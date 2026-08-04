import os
import shutil
import argparse
import pickle
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from disorderflow.datasets import get_dataset
from disorderflow.models import get_model
from disorderflow.utils.misc import *
from disorderflow.utils.data import *
from disorderflow.utils.train import *


class ScaffoldGroupedBatchSampler(torch.utils.data.Sampler):
    """Samples batches where each batch groups variants from the same scaffold.

    This is essential for V14 grouped losses (conf_variance, conf_ranking,
    grouped_margin) that compare different CDR designs WITHIN the same scaffold.

    Each batch draws `batch_size` entries that share the same scaffold_id.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        scaffold_ids = getattr(dataset, '_scaffold_ids', None)
        groups = {}  # scaffold_id -> list of indices
        for idx in range(len(dataset)):
            sid = scaffold_ids[idx] if scaffold_ids else idx
            groups.setdefault(sid, []).append(idx)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = []

        for sid, indices in groups.items():
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                if len(batch) > 1:
                    self.batches.append(batch)

        # Remaining singletons (1 variant/scaffold or leftovers) grouped together
        singles = []
        for sid, indices in groups.items():
            if len(indices) == 1:
                singles.append(indices[0])
        while len(singles) >= batch_size:
            self.batches.append(singles[:batch_size])
            singles = singles[batch_size:]

        if not self.batches:
            for sid, indices in groups.items():
                for i in indices:
                    self.batches.append([i])

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


if __name__ == '__main__':
    # Fix for macOS shared memory issue
    import torch.multiprocessing
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tag', type=str, default='')
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument('--resume', type=str, default=None)
    checkpoint_group.add_argument('--finetune', type=str, default=None)
    checkpoint_group.add_argument(
        '--init', type=str, default=None,
        help='Load every shape-compatible model tensor and start a fresh training stage')
    parser.add_argument('--accum_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--max-iters', type=int, default=None, help='Override config max_iters')
    parser.add_argument('--val-freq', type=int, default=None, help='Override config val_freq')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    if args.max_iters is not None:
        config.train.max_iters = args.max_iters
    if args.val_freq is not None:
        config.train.val_freq = args.val_freq
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(f"Loss weights: {config.train.loss_weights}")
    logger.info(config)

    # Data
    logger.info('Loading dataset...')
    train_dataset = get_dataset(config.dataset.train)
    val_dataset = get_dataset(config.dataset.val)

    # WAY4 V7 P0-fix-B: wrap dataset with per-residue disorder profiles
    disorder_lookup_path = config.dataset.get('disorder_lookup') if hasattr(config, 'dataset') else None
    split_lookup_paths = {
        'train': config.dataset.get('disorder_lookup_train'),
        'val': config.dataset.get('disorder_lookup_val'),
    } if hasattr(config, 'dataset') else {}
    split_lookup_labels = {
        'train': config.dataset.get('disorder_lookup_train_split', 'train'),
        'val': config.dataset.get('disorder_lookup_val_split', 'val'),
    } if hasattr(config, 'dataset') else {}
    if disorder_lookup_path is not None or any(split_lookup_paths.values()):
        from disorderflow.datasets.disorder_augmented import DisorderAugmentedDataset, load_disorder_lookup
        # Wrap both splits so validation measures the same conditioning task.
        for split_name, dataset in [('train', train_dataset), ('val', val_dataset)]:
            ids = getattr(dataset, 'ids', getattr(dataset, 'all_ids', None))
            if ids is None:
                logger.warning(
                    'Disorder lookup provided but %s sample IDs are unavailable; skipping',
                    split_name)
                continue
            lookup_path = split_lookup_paths.get(split_name) or disorder_lookup_path
            if lookup_path is None:
                raise RuntimeError(f'Missing disorder lookup for {split_name} split')
            disorder_lookup = load_disorder_lookup(
                lookup_path,
                expected_split=split_lookup_labels[split_name]
                if split_lookup_paths.get(split_name) else None,
                expected_ids=ids if split_lookup_paths.get(split_name) else None,
                require_envelope=bool(split_lookup_paths.get(split_name)),
            )
            if config.dataset.get('require_disorder_lookup', False):
                valid_ids = {
                    str(key).casefold() for key, value in disorder_lookup.items()
                    if value is not None
                }
                filtered_ids = [
                    value for value in ids if str(value).casefold() in valid_ids
                ]
                if not filtered_ids:
                    raise RuntimeError(
                        f'No {split_name} samples have required disorder profiles')
                dataset.ids = filtered_ids
                ids = filtered_ids
                logger.info(
                    'Filtered %s to %d samples with measured disorder profiles',
                    split_name, len(ids))
            wrapped = DisorderAugmentedDataset(dataset, disorder_lookup, ids)
            if split_name == 'train':
                train_dataset = wrapped
            else:
                val_dataset = wrapped
            logger.info(
                'Disorder-augmented %s split: %d profiles for %d samples',
                split_name, len(disorder_lookup), len(ids))

    if config.dataset.get('contrastive_cross_sample_negatives', False):
        from disorderflow.datasets.disorder_augmented import ContrastiveNegativeDataset
        train_dataset = ContrastiveNegativeDataset(train_dataset)
        val_dataset = ContrastiveNegativeDataset(val_dataset)
        logger.info('Enabled cross-sample CDR negatives for train and validation')
    
    # Custom Sampler for CDR-type consistency
    if getattr(train_dataset, 'requires_complete_groups', False):
        batch_sampler = CompleteGroupBatchSampler(
            train_dataset, config.train.batch_size, shuffle=True,
            seed=config.train.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=PaddingCollate(),
            num_workers=args.num_workers,
        )
    elif config.dataset.train.type == 'lmdb_preprocessed':
        db_dir = os.path.dirname(config.dataset.train.db_path)
        indices_path = os.path.join(db_dir, 'meta_indices.pkl')
        with open(indices_path, 'rb') as f:
            indices_by_type = pickle.load(f)
        batch_sampler = CDRBatchSampler(indices_by_type, config.train.batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=PaddingCollate(), 
            num_workers=args.num_workers
        )
    else:
        # Use grouped batch sampler if dataset has scaffold_ids (V14+)
        scaffold_ids = getattr(train_dataset, '_scaffold_ids', None)
        if scaffold_ids is not None:
            batch_sampler = ScaffoldGroupedBatchSampler(
                train_dataset, config.train.batch_size, shuffle=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=PaddingCollate(),
                num_workers=args.num_workers
            )
        else:
            # Use WeightedRandomSampler if dataset provides sample weights (e.g., IDP oversampling)
            sample_weights = getattr(train_dataset, '_sample_weights', None)
            if sample_weights is not None:
                from torch.utils.data import WeightedRandomSampler
                sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.train.batch_size,
                    collate_fn=PaddingCollate(),
                    sampler=sampler,
                    num_workers=args.num_workers
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.train.batch_size,
                    collate_fn=PaddingCollate(),
                    shuffle=True,
                    num_workers=args.num_workers
                )
        
    train_iterator = inf_iterator(train_loader)
    if getattr(val_dataset, 'requires_complete_groups', False):
        val_batch_sampler = CompleteGroupBatchSampler(
            val_dataset, config.train.batch_size, shuffle=False,
            seed=config.train.seed)
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=PaddingCollate(),
            num_workers=args.num_workers,
        )
    else:
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    amp_dtype_name = str(config.train.get('amp_dtype', 'float16')).lower()
    amp_dtypes = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if amp_dtype_name not in amp_dtypes:
        raise ValueError(f'Unsupported AMP dtype: {amp_dtype_name}')
    amp_dtype = amp_dtypes[amp_dtype_name]
    amp_enabled = args.device not in ('cpu', 'mps') and not args.no_amp
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    logger.info('AMP enabled=%s dtype=%s grad_scaler=%s',
                amp_enabled, amp_dtype, scaler_enabled)
    scaler = torch.amp.GradScaler(
        'xpu' if args.device == 'xpu' else 'cuda',
        enabled=scaler_enabled,
        init_scale=128.0)
    optimizer.zero_grad()
    it_first = 1
    min_val_loss = float('inf')

    # Resume or Finetune
    if args.resume is not None:
        # Resume: restore everything (model, optimizer, scheduler, iteration)
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        it_first = ckpt['iteration'] + 1
        min_val_loss = ckpt.get('min_val_loss', float('inf'))
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
        if missing_keys:
            logger.info(f'New parameters (random init): {len(missing_keys)}')
            for k in missing_keys[:8]:
                logger.info(f'  + {k}')
        if unexpected_keys:
            logger.info(f'Removed parameters (ignored): {len(unexpected_keys)}')
        # Optimizer state may be incompatible when model has new parameters.
        # Fall back to fresh optimizer if sizes don't match.
        try:
            logger.info('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        except ValueError:
            logger.info('Optimizer state incompatible (new params added) — using fresh optimizer.')
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
    elif args.finetune is not None:
        # Finetune: only load model weights, start fresh (new optimizer, scheduler, warmup)
        logger.info('Finetuning from checkpoint: %s' % args.finetune)
        ckpt = torch.load(args.finetune, map_location=args.device, weights_only=False)
        # Strip old confidence head weights only if architecture differs from checkpoint.
        # V9+ checkpoints have the current head architecture; older checkpoints
        # (V8 and before) have incompatible head shapes. We detect this by checking
        # for the old 'pair_proj' key which was replaced by AttentionPAEHead.
        ckpt_state = ckpt['model']
        # V18: always strip architecture-dependent heads when finetuning
        # head_seq changed shape (768→1024), disorder_proj is new
        old_head_names = ['head_plddt.', 'head_iptm.', 'head_pae.', 'pair_proj.',
                          'head_seq.', 'head_seq_fixbb.', 'disorder_proj.', 'head_pos.', 'head_ori.', 'head_ang.']
        stripped_keys = []
        for k in list(ckpt_state.keys()):
            if any(name in k for name in old_head_names):
                stripped_keys.append(k)
                ckpt_state.pop(k)
        if stripped_keys:
            logger.info(f'Stripped {len(stripped_keys)} old head weights (new heads randomized)')
            for k in stripped_keys[:5]:
                logger.info(f'  - {k}')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
        if missing_keys:
            logger.info(f'New parameters (random init): {len(missing_keys)}')
            for k in missing_keys[:10]:
                logger.info(f'  + {k}')
        if unexpected_keys:
            logger.info(f'Deprecated parameters (skipped): {len(unexpected_keys)}')
        logger.info('Loaded model weights. Starting fresh with new optimizer/scheduler.')
        # it_first and min_val_loss stay at default (1 and inf)
    elif args.init is not None:
        # Retain compatible heads learned by the previous curriculum stage.
        logger.info('Initializing compatible weights from checkpoint: %s' % args.init)
        ckpt = torch.load(args.init, map_location=args.device, weights_only=False)
        source_state = ckpt['model']
        target_state = model.state_dict()
        compatible = {
            key: value for key, value in source_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        shape_mismatches = [
            key for key, value in source_state.items()
            if key in target_state and target_state[key].shape != value.shape
        ]
        missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)
        loaded_numel = sum(value.numel() for value in compatible.values())
        total_numel = sum(value.numel() for value in target_state.values())
        logger.info(
            'Compatible initialization loaded %d/%d tensors (%.1f%% of parameters)',
            len(compatible), len(target_state), 100.0 * loaded_numel / total_numel)
        if shape_mismatches:
            logger.info('Shape-mismatched parameters skipped: %d', len(shape_mismatches))
        if missing_keys:
            logger.info('New parameters (random init): %d', len(missing_keys))
            for key in missing_keys[:10]:
                logger.info('  + %s', key)
        if unexpected_keys:
            logger.info('Deprecated parameters (skipped): %d', len(unexpected_keys))

    # Freeze backbone for confidence head fine-tuning
    if config.train.get('freeze_backbone', False):
        logger.info('Freezing backbone for confidence head fine-tuning...')
        trainable_suffixes = config.train.get('trainable_modules', [
            'head_plddt',
            'head_iptm',
            'head_pae',
            'head_disorder',
            'conf_embed',
            'iptm_embed',
            'pae_embed',
            'pair_proj',
            'v12_plddt',      # V12 pure-sequence confidence heads
            'v12_iptm',
            'v12_pae',
            'v12_seq_emb',
            'v13_seq_feedback',  # V13 sequence embedding feedback
            'head_seq',        # V7+ CDR sequence head (disorder-conditioned)
            'head_seq_fixbb',  # V7+ fixbb fallback head
            'disorder_proj',   # V7+ epitope disorder projection
            'head_pos',        # V7+ position head
            'head_ori',        # V7+ orientation head
            'head_ang',        # V7+ angle head
        ])
        # Optionally unfreeze the last N encoder layers (GAEncoder blocks)
        unfreeze_encoder_layers = config.train.get('unfreeze_encoder_layers', 0)
        if unfreeze_encoder_layers > 0:
            logger.info(f'Also unfreezing last {unfreeze_encoder_layers} encoder layers...')

        for name, param in model.named_parameters():
            if any(suffix in name for suffix in trainable_suffixes):
                param.requires_grad = True
            elif unfreeze_encoder_layers > 0:
                # Check if this param belongs to the last N encoder blocks
                should_unfreeze = False
                for layer_idx in range(6 - unfreeze_encoder_layers, 6):
                    if f'encoder.blocks.{layer_idx}' in name:
                        should_unfreeze = True
                        break
                param.requires_grad = should_unfreeze
            else:
                param.requires_grad = False

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f'Trainable: {n_trainable:,}/{n_total:,} ({100*n_trainable/n_total:.1f}%)')

    # EMA
    ema_decay = config.train.get('ema_decay', 0.0)
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        logger.info(f'EMA enabled with decay={ema_decay}')

    # Negative sample augmentation for confidence calibration (Phase 4)
    # Randomly scramble sequences to create "bad design" examples,
    # forcing confidence heads to learn low-confidence predictions.
    def neg_sample_aug(batch, neg_prob, neg_confidence, neg_scramble_ratio):
        """In-place batch modification for negative sample training.

        neg_prob: probability of converting a sample to negative
        neg_confidence: dict of low confidence values {plddt, iptm, pae}
        neg_scramble_ratio: [min, max] fraction of sequence to scramble
        """
        if neg_prob <= 0 or torch.rand(1).item() > neg_prob:
            return

        aa = batch['aa']  # (N, L)
        mask = batch['mask']  # (N, L)
        N, L = aa.shape

        for b in range(N):
            valid_len = mask[b].sum().int().item()
            if valid_len < 4:
                continue
            ratio_min, ratio_max = neg_scramble_ratio
            ratio = ratio_min + (ratio_max - ratio_min) * torch.rand(1).item()
            scramble_len = max(2, int(valid_len * ratio))
            start = torch.randint(0, max(1, valid_len - scramble_len), (1,)).item()
            end = start + scramble_len

            # Scramble amino acids in the region
            random_aa = torch.randint(0, 20, (scramble_len,), device=aa.device)
            aa[b, start:end] = random_aa

            # Modify ground truth: low confidence for scrambled regions
            neg_plddt = neg_confidence.get('plddt', 0.1)
            neg_iptm = neg_confidence.get('iptm', 0.05)
            neg_pae_norm = neg_confidence.get('pae', 0.65)  # ~20.0 / 31.0

            if 'af2_plddt' in batch:
                batch['af2_plddt'][b, start:end] = neg_plddt
            if 'af2_iptm' in batch:
                batch['af2_iptm'][b] = neg_iptm
            if 'af2_pae_matrix' in batch:
                pae_mat = batch['af2_pae_matrix']
                pae_mat[b, start:end, :] = neg_pae_norm
                pae_mat[b, :, start:end] = neg_pae_norm
            # NOTE: We do NOT modify disorder_label for negative samples.
            # The structure is still correct (only sequence is scrambled),
            # so the disorder state of the structure is unchanged.
            # Only confidence labels (pLDDT/ipTM/PAE) should change because
            # they measure sequence-structure compatibility.

    neg_prob = config.train.get('neg_sample_prob', 0.0)
    neg_confidence = config.train.get('neg_confidence', {'plddt': 0.1, 'iptm': 0.05, 'pae': 0.65})
    neg_scramble_ratio = config.train.get('neg_scramble_ratio', [0.2, 0.5])

    # Train
    consecutive_failed_steps = [0]

    def train(it):
        time_start = current_milli_time()
        model.train()

        accum_loss_dict = {}
        valid_batch_count = 0

        # Gradient Accumulation
        for i in range(args.accum_steps):
            # Prepare data
            try:
                batch = recursive_to(next(train_iterator), args.device)
            except StopIteration:
                break

            # Negative sample augmentation (Phase 4)
            neg_sample_aug(batch, neg_prob, neg_confidence, neg_scramble_ratio)

            if 'fixed_t' in config.train:
                batch['fixed_t'] = config.train.fixed_t

            # Forward
            # if args.debug: torch.set_anomaly_enabled(True)
            device_type = args.device if args.device in ('cuda', 'xpu') else ('mps' if args.device == 'mps' else 'cpu')
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = model(batch)
                avg_t = loss_dict.pop('avg_t', 0.0)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                # IDP-aware loss weighting: multiply loss for IDP entries
                # to force model to pay attention to disordered regions
                idp_mult = config.train.get('idp_loss_multiplier', 1.0)
                if idp_mult != 1.0:
                    batch_is_idp = batch.get('is_idp', None)
                    if batch_is_idp is not None and batch_is_idp.any():
                        loss = loss * idp_mult
                loss_dict['overall'] = loss
                loss_dict['avg_t'] = avg_t # Put it back for logging

            # NaN Check and Skip
            if not torch.isfinite(loss):
                logger.warning(f'NaN or Inf detected in loss at iter {it}, micro-batch {i}. Skipping micro-batch.')
                logger.warning(f'Loss dict: {loss_dict}')
                continue

            # Normalize loss by accumulation steps
            loss = loss / args.accum_steps

            # Backward
            scaler.scale(loss).backward()
            
            # Accumulate logging stats
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                accum_loss_dict[k] = accum_loss_dict.get(k, 0.0) + val
            valid_batch_count += 1

        time_forward_end = current_milli_time()

        if valid_batch_count == 0:
            logger.warning(f'All micro-batches failed at iter {it}. Skipping step.')
            optimizer.zero_grad()
            consecutive_failed_steps[0] += 1
            max_failed = int(config.train.get('max_consecutive_failed_steps', 3))
            if consecutive_failed_steps[0] >= max_failed:
                raise RuntimeError(
                    f'{consecutive_failed_steps[0]} consecutive training steps had no finite micro-batches')
            return

        consecutive_failed_steps[0] = 0

        # Average logging stats
        for k in accum_loss_dict:
            accum_loss_dict[k] /= valid_batch_count

        # Logging
        # Extract average t for logging
        avg_t = accum_loss_dict.pop('avg_t', 0.0)
        
        # Convert losses to tensors for log_losses compatibility
        for k in accum_loss_dict:
            accum_loss_dict[k] = torch.tensor(accum_loss_dict[k])

        # Optimizer Step
        scaler.unscale_(optimizer)
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        # Skip step if gradients overflowed (FP16 can produce NaN/Inf grads
        # even with finite loss, especially early in training)
        if not torch.isfinite(orig_grad_norm):
            logger.warning(f'Iter {it}: NaN/Inf gradients detected, skipping optimizer step')
            scaler.update()
            optimizer.zero_grad()
            return

        # Linear Warmup
        warmup_steps = config.train.get('warmup_steps', 1000)
        if it <= warmup_steps:
            warmup_factor = float(it) / float(warmup_steps)
            for param_group in optimizer.param_groups:
                # Assuming the initial LR in config is the target LR
                # We need to store the base LR somewhere to be cleaner, 
                # but for now we assume config.train.optimizer.lr is the target.
                param_group['lr'] = config.train.optimizer.lr * warmup_factor

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if ema is not None:
            ema.update(model)

        time_backward_end = current_milli_time()

        log_freq = config.train.get('log_freq', 1)
        if it % log_freq == 0:
            log_losses(accum_loss_dict, it, 'train', logger, writer, others={
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                't': avg_t,
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })

    # Validate
    def validate(it):
        if ema is not None:
            ema.apply(model)

        loss_tape = ValidationLossTape()
        loss_tape_idp = ValidationLossTape()
        loss_tape_folded = ValidationLossTape()
        n_idp = 0
        n_folded = 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                batch['fixed_t'] = config.train.get('val_fixed_t', 0.5)
                if config.model.loss_weight.get('contrastive', 0) > 0:
                    batch['contrastive_corrupt'] = torch.tensor(
                        [bool(i % 2)] * batch['aa'].shape[0], device=args.device)
                validation_seed = int(config.train.get(
                    'validation_seed', config.train.seed)) + i
                torch.manual_seed(validation_seed)
                if args.device == 'cuda':
                    torch.cuda.manual_seed_all(validation_seed)
                elif args.device == 'xpu' and hasattr(torch.xpu, 'manual_seed_all'):
                    torch.xpu.manual_seed_all(validation_seed)
                # Forward
                device_type = 'cuda' if args.device == 'cuda' else 'cpu'
                if args.device == 'mps': device_type = 'mps'
                with torch.autocast(
                        device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    loss_dict = model(batch)
                    avg_t = loss_dict.pop('avg_t', 0.0)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_dict['avg_t'] = avg_t

                # Skip NaN val samples (same logic as training micro-batch NaN skip)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss_tape.update(loss_dict, 1)
                # Split tracking: IDP vs folded
                batch_is_idp = batch.get('is_idp', None)
                is_idp = batch_is_idp is not None and batch_is_idp.any().item()
                if is_idp:
                    loss_tape_idp.update(loss_dict, 1)
                    n_idp += 1
                else:
                    loss_tape_folded.update(loss_dict, 1)
                    n_folded += 1

        avg_loss = loss_tape.log(it, logger, writer, 'val')
        if n_idp > 0:
            loss_tape_idp.log(it, logger, writer, 'val/idp')
        if n_folded > 0:
            loss_tape_folded.log(it, logger, writer, 'val/folded')
        # Don't step scheduler during warmup —warmup manually controls LR.
        # Stepping during warmup can cause the scheduler to decay its internal LR,
        # leading to a sudden drop when warmup ends and the scheduler takes over.
        if it > config.train.get('warmup_steps', 0):
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        selection_key = config.train.get('checkpoint_selection_metric')
        if selection_key:
            if selection_key not in loss_tape.accumulate:
                raise RuntimeError(
                    f'Checkpoint selection metric missing from validation: {selection_key}')
            selection_value = loss_tape.accumulate[selection_key] / loss_tape.total
            logger.info(
                'Checkpoint selection metric %s=%.6f',
                selection_key, float(selection_value))
        else:
            selection_value = avg_loss
        max_seq_loss = config.train.get('checkpoint_selection_max_seq_loss')
        if max_seq_loss is not None:
            seq_value = loss_tape.accumulate['seq'] / loss_tape.total
            if seq_value > float(max_seq_loss):
                logger.info(
                    'Checkpoint rejected: seq loss %.6f exceeds %.6f',
                    float(seq_value), float(max_seq_loss))
                selection_value = float('inf')

        if ema is not None:
            ema.restore(model)

        return selection_value

    # Early stopping
    early_stop_patience = config.train.get('early_stopping_patience', 0)
    early_stop_counter = 0

    try:
        for it in range(it_first, config.train.max_iters + 1):
            # Length curriculum: progressively increase max_residues during training
            length_curriculum = config.train.get('length_curriculum', None)
            if length_curriculum is not None:
                stages = length_curriculum.get('stages', [])
                for stage_iter, stage_max_len in stages:
                    if it == stage_iter:
                        logger.info(f'[LengthCurriculum] iter {it}: max_residues -> {stage_max_len}')
                        if hasattr(train_dataset, 'set_max_residues'):
                            train_dataset.set_max_residues(stage_max_len)
                        if hasattr(val_dataset, 'set_max_residues'):
                            val_dataset.set_max_residues(stage_max_len)
            train(it)
            if it % config.train.val_freq == 0:
                avg_val_loss = validate(it)
                is_best = avg_val_loss < min_val_loss
                if is_best:
                    min_val_loss = avg_val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if not args.debug:
                    ckpt_state = {
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'iteration': it,
                        'avg_val_loss': avg_val_loss,
                        'min_val_loss': min_val_loss,
                        'weights_kind': 'raw',
                    }
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save(ckpt_state, ckpt_path)
                    if is_best:
                        best_path = os.path.join(ckpt_dir, 'best.pt')
                        best_state = dict(ckpt_state)
                        if ema is not None:
                            best_state['model'] = ema.averaged_state_dict(model)
                            best_state['weights_kind'] = 'ema'
                        torch.save(best_state, best_path)
                        logger.info(f'New best model saved to {best_path} with loss {min_val_loss:.4f}')
                # Early stopping check
                if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                    logger.info(
                        f'Early stopping: no improvement for {early_stop_patience} '
                        f'validations (best={min_val_loss:.4f})'
                    )
                    break
    except KeyboardInterrupt:
        logger.info('Terminating...')
    finally:
        if not args.debug:
            writer.close()
        # Kill dataloader workers
        if 'train_iterator' in locals():
            del train_iterator
        logger.info('Done.')
