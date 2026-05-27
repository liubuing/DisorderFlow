import os
import shutil
import argparse
import pickle
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from antibodydesignbfn.datasets import get_dataset
from antibodydesignbfn.models import get_model
from antibodydesignbfn.utils.misc import *
from antibodydesignbfn.utils.data import *
from antibodydesignbfn.utils.train import *


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
    parser.add_argument('--device', type=str, default='xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)
    parser.add_argument('--accum_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
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
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
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
    
    # Custom Sampler for CDR-type consistency
    if config.dataset.train.type == 'lmdb_preprocessed':
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
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, collate_fn=PaddingCollate(), shuffle=False, num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    scaler = torch.amp.GradScaler('xpu' if args.device == 'xpu' else 'cuda', enabled=(args.device not in ('cpu', 'mps') and not args.no_amp), init_scale=128.0)
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
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
    elif args.finetune is not None:
        # Finetune: only load model weights, start fresh (new optimizer, scheduler, warmup)
        logger.info('Finetuning from checkpoint: %s' % args.finetune)
        ckpt = torch.load(args.finetune, map_location=args.device, weights_only=False)
        ckpt_state = ckpt['model']

        # Strip old confidence head weights (architecture may differ).
        # Use substring match because checkpoint keys include module prefix
        # (e.g. bfn.receiver.head_plddt.0.weight).
        old_head_names = ['head_plddt.', 'head_iptm.', 'head_pae.', 'pair_proj.']
        stripped_keys = []
        for k in list(ckpt_state.keys()):
            if any(name in k for name in old_head_names):
                stripped_keys.append(k)
                ckpt_state.pop(k)

        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
        if stripped_keys:
            logger.info(f'Stripped {len(stripped_keys)} old confidence head weights (new heads randomized)')
            for k in stripped_keys[:5]:
                logger.info(f'  - {k}')
        if missing_keys:
            logger.info(f'New parameters (random init): {len(missing_keys)}')
            for k in missing_keys[:10]:
                logger.info(f'  + {k}')
        if unexpected_keys:
            logger.info(f'Deprecated parameters (skipped): {len(unexpected_keys)}')
        logger.info('Loaded model weights. Starting fresh with new optimizer/scheduler.')
        # it_first and min_val_loss stay at default (1 and inf)

    # Freeze backbone for confidence head fine-tuning
    if config.train.get('freeze_backbone', False):
        logger.info('Freezing backbone for confidence head fine-tuning...')
        trainable_suffixes = [
            'head_plddt',
            'head_iptm',
            'head_pae',
            'conf_embed',
            'iptm_embed',
            'pae_embed',
            'pair_proj',
        ]
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

    # Train
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

            if 'fixed_t' in config.train:
                batch['fixed_t'] = config.train.fixed_t

            # Forward
            # if args.debug: torch.set_anomaly_enabled(True)
            device_type = args.device if args.device in ('cuda', 'xpu') else ('mps' if args.device == 'mps' else 'cpu')
            use_amp = args.device not in ('cpu', 'mps') and not args.no_amp

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
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
            return

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
                # Forward
                device_type = 'cuda' if args.device == 'cuda' else 'cpu'
                if args.device == 'mps': device_type = 'mps'
                use_amp = args.device != 'cpu' and not args.no_amp and args.device != 'mps'
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                    loss_dict = model(batch)
                    avg_t = loss_dict.pop('avg_t', 0.0)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_dict['avg_t'] = avg_t

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
        # Don't step scheduler during warmup — warmup manually controls LR.
        # Stepping during warmup can cause the scheduler to decay its internal LR,
        # leading to a sudden drop when warmup ends and the scheduler takes over.
        if it > config.train.get('warmup_steps', 0):
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        return avg_loss

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
                    }
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save(ckpt_state, ckpt_path)
                    if is_best:
                        best_path = os.path.join(ckpt_dir, 'best.pt')
                        torch.save(ckpt_state, best_path)
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
