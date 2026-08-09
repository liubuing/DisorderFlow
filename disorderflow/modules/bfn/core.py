import torch
import torch.nn as nn
import torch.nn.functional as F
from disorderflow.modules.bfn.sequence import CategoricalBFN
from disorderflow.modules.bfn.position import PositionBFN
from disorderflow.modules.bfn.orientation import OrientationBFN
from disorderflow.modules.bfn.sidechain import SidechainBFN
from disorderflow.modules.bfn.receiver import AntibodyBFN_Receiver
from disorderflow.modules.bfn.core_losses import (
    within_group_pair_mask as _within_group_pair_mask,
    grouped_std as _grouped_std,
    grouped_margin_ranking as _grouped_margin_ranking,
    within_protein_disorder_ranking as _within_protein_disorder_ranking,
)
from disorderflow.modules.bfn.contrastive_losses import grouped_contrastive_margin
from disorderflow.modules.common.geometry import (
    construct_3d_basis,
    repr_6d_to_rotation_matrix,
    compute_fape,
)
from disorderflow.modules.common.ot_align import LieOTAlign
from disorderflow.utils.protein.constants import Fragment


def sequence_cross_entropy_20(logits, targets, label_smoothing=0.0):
    """Per-residue CE over the 20 valid amino-acid classes only."""
    return F.cross_entropy(
        logits[..., :20].float().reshape(-1, 20), targets.flatten(),
        reduction='none', ignore_index=-100, label_smoothing=label_smoothing,
    ).reshape(targets.shape)


class AntibodyBFN_Core(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, num_steps, eps_net_opt={}, position_mean=[0.0, 0.0, 0.0], position_scale=[10.0], loss_weight={}, ot_opt={}, beta=1.0, schedule='linear'):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta  # Configurable BFN precision
        self.schedule = schedule  # 'linear' or 'cosine'
        self.num_classes = 22
        
        # Use only explicitly defined loss weights from config
        self.loss_weight = loss_weight.copy() if loss_weight else {}
        
        # Pass beta and schedule to flow components
        self.flow_seq = CategoricalBFN(num_classes=self.num_classes, num_steps=num_steps, beta=beta, schedule=schedule)
        self.flow_pos = PositionBFN()
        self.flow_ori = OrientationBFN()
        self.flow_ang = SidechainBFN()
        
        # Detect seq-only mode: true when no structure losses are present.
        # Confidence losses (pLDDT/ipTM/PAE) don't need structure prediction —
        # the receiver should use the true backbone positions and predict confidence.
        conf_loss_keys = ['plddt', 'iptm', 'pae']
        has_structure_loss = any(k in loss_weight and loss_weight.get(k, 0) > 0 for k in ['dist', 'fape', 'pos', 'rot'])
        has_conf_loss = any(k in loss_weight and loss_weight.get(k, 0) > 0 for k in conf_loss_keys)
        seq_only = not has_structure_loss
        head_dropout = loss_weight.get('head_dropout', 0.1) if loss_weight else 0.1
        disorder_head = (loss_weight.get('disorder', 0) > 0
                         or loss_weight.get('disorder_rank', 0) > 0) if loss_weight else False
        encoder_dropout = loss_weight.get('encoder_dropout', 0.0) if loss_weight else 0.0
        self.receiver = AntibodyBFN_Receiver(res_feat_dim, pair_feat_dim, num_layers=6, encoder_opt=eps_net_opt, num_classes=self.num_classes, seq_only=seq_only, head_dropout=head_dropout, disorder_head=disorder_head, encoder_dropout=encoder_dropout)
        
        ot_iters = ot_opt.get('num_iters', 10)
        ot_eps = ot_opt.get('epsilon', 2.0)
        self.ot_align = LieOTAlign(num_iters=ot_iters, epsilon=ot_eps)
        
        self.register_buffer('_dummy', torch.empty([0, ]))
        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _mask_antigen(self, batch, mask_gen):
        """Antigen-context mask: real residues on chains other than the scaffold.
        Scaffold chain = the chain_nb that bears any generate_flag (CDR) residue.
        Antigen = real residues with a different chain_nb. Empty when there is
        only one chain (fixbb → no antigen visible). Drives direction D's
        antigen→CDR cross-attention."""
        fragment_type = batch.get('fragment_type')
        if fragment_type is not None:
            return batch['mask'].bool() & (fragment_type == int(Fragment.Antigen))
        cn = batch.get('chain_nb')
        if cn is None:
            return None
        device = mask_gen.device
        N, L = mask_gen.shape
        out = torch.zeros(N, L, dtype=torch.bool, device=device)
        for b in range(N):
            cdr_idx = mask_gen[b].bool()
            if cdr_idx.any() and cn is not None:
                # scaffold chain nb = chain_nb at the first CDR residue
                cnd = cn[b]
                if cnd.numel() >= L:
                    scaff_nb = int(cnd[cdr_idx][0].item())
                    real = batch['mask'][b].bool()
                    out[b] = real & (cnd[:L] != scaff_nb)
        return out

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, batch):
        """
        Training step.
        """
        # Data
        x_seq_target = batch['aa'].clone() # (N, L), immutable supervision target
        x_seq = x_seq_target.clone() # receiver input may be contrastively augmented
        x_pos = batch['pos_heavyatom'][:, :, 1] # CA (N, L, 3)
        cb = batch['pos_heavyatom'][:, :, 4]
        cb_mask = batch['mask_heavyatom'][:, :, 4].bool()
        contact_pos = torch.where(cb_mask.unsqueeze(-1), cb, x_pos)
        x_pos = self._normalize_position(x_pos) # Normalize position
        
        # === Backbone atom coordinates (N, CA, C, O) ===
        # These are given directly to help model learn geometric features
        # Index: 0=N, 1=CA, 2=C, 3=O
        backbone_pos = batch['pos_heavyatom'][:, :, :4]  # (N, L, 4, 3)
        
        # Construct Rotation
        from disorderflow.modules.common.geometry import construct_3d_basis
        R_true = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1], # CA
            batch['pos_heavyatom'][:, :, 2], # C
            batch['pos_heavyatom'][:, :, 0], # N
        ) # (N, L, 3, 3)
        x_ori = R_true
        
        x_ang = batch['torsion'] # (N, L, 4) - sidechain chi angles
        mask_res = batch['mask']
        mask_gen = batch['generate_flag'].clone()

        # Direction D: build mask_antigen (residues on chains OTHER than the
        # scaffold, i.e. the antigen context visible in complex mode). The
        # scaffold chain is the one bearing generate_flag residues; antigen =
        # real residues on different chain_nb. Empty in fixbb (single chain).
        batch['mask_antigen'] = self._mask_antigen(batch, mask_gen)

        # === Confidence Mask Augmentation ===
        # Randomly set generate_flag=1 for a fraction of residues to simulate
        # design-time partial information conditions. This forces the ipTM head
        # to predict confidence from incomplete structural context, closing the
        # train/design distribution gap.
        N_batch = x_seq.size(0)  # needed for mask augmentation below
        mask_aug_prob = self.loss_weight.get('mask_aug_prob', 0.0)
        if self.training and mask_aug_prob > 0 and torch.rand(1).item() < mask_aug_prob:
            mask_aug_ratio = self.loss_weight.get('mask_aug_ratio', [0.2, 1.0])
            ratio = mask_aug_ratio[0] + (mask_aug_ratio[1] - mask_aug_ratio[0]) * torch.rand(1).item()
            # Only augment real, non-padding residues that aren't already masked
            real_mask = mask_res.bool() & (~mask_gen.bool())
            for b in range(N_batch):
                real_idx = torch.where(real_mask[b])[0]
                if len(real_idx) > 1:
                    n_m = max(1, int(len(real_idx) * ratio))
                    n_m = min(n_m, len(real_idx))
                    chosen = real_idx[torch.randperm(len(real_idx))[:n_m]]
                    mask_gen[b, chosen] = True

        # Direction H: contrastive CDR-antigen matching (§2.2).
        # Shuffle CDR within mask_gen so the model sees fake CDR+antigen pairs.
        # The contrastive head (receiver) predicts real=1 / shuffled=0,
        # forcing the encoder to learn antigen-compatible CDR features.
        N_batch = x_seq.size(0)
        contrastive_w = self.loss_weight.get('contrastive', 0)
        contrastive_target = torch.ones(N_batch, device=x_seq.device)
        contrastive_valid = torch.zeros(N_batch, dtype=torch.bool, device=x_seq.device)
        if contrastive_w > 0 and mask_gen.any() and batch.get('mask_antigen') is not None and batch['mask_antigen'].any():
            for b in range(N_batch):
                cdr_idx = torch.where(mask_gen[b].bool())[0]
                has_antigen = batch['mask_antigen'][b].any()
                contrastive_valid[b] = has_antigen and len(cdr_idx) > 0
                forced_corruption = batch.get('contrastive_corrupt', False)
                if isinstance(forced_corruption, torch.Tensor):
                    forced_corruption = bool(forced_corruption.flatten()[b].item())
                should_corrupt = (self.training and torch.rand(1).item() < 0.5) or forced_corruption
                if has_antigen and len(cdr_idx) >= 2 and should_corrupt:
                    negative_aa = batch.get('contrastive_negative_aa', None)
                    if negative_aa is not None:
                        x_seq[b, cdr_idx] = negative_aa[b, cdr_idx]
                    else:
                        perm = torch.randperm(len(cdr_idx), device=x_seq.device)
                        x_seq[b, cdr_idx] = x_seq[b, cdr_idx[perm]]
                    contrastive_target[b] = 0.0  # this entry is now fake

        # 1. Sample Time t
        N = N_batch
        if 'fixed_t' in batch and batch['fixed_t'] is not None:
            t = torch.full((N,), batch['fixed_t'], device=x_seq.device)
        else:
            t_clamp_max = self.loss_weight.get('t_clamp', 1.0)
            t = torch.rand(N, device=x_seq.device).clamp(1e-5, t_clamp_max)

        # 2. Sample Theta (Parameters) from Clean Data
        t_expanded = t[:, None].expand(N, x_seq.size(1))

        theta_seq = self.flow_seq.sample_theta(x_seq, t_expanded)
        theta_pos = self.flow_pos.sample_theta(x_pos, t_expanded)
        theta_ori = self.flow_ori.sample_theta(x_ori, t_expanded)
        theta_ang_full = self.flow_ang.sample_theta(x_ang, t_expanded)
        theta_ang = self.flow_ang.get_angle(theta_ang_full)
        
        # 3. Receiver (Predict Clean Data)
        pair_feat = batch['pair_feat']
        # Zero pair features for mask-augmented residues so the encoder
        # sees no structural pair information for them (matches design setting).
        if mask_aug_prob > 0 and mask_gen.any():
            aug_mask = mask_gen.bool() & ~batch['generate_flag'].bool()
            if aug_mask.any():
                pair_feat = pair_feat.clone()
                for b in range(N):
                    idx = torch.where(aug_mask[b])[0]
                    if len(idx) > 0:
                        pair_feat[b, idx, :, :] = 0
                        pair_feat[b, :, idx, :] = 0
        
        # Convert theta to estimated mean/clean input for receiver
        # This is critical! theta scales with alpha(t), but receiver expects x_hat.
        inp_seq = self.flow_seq.probabilities(theta_seq)
        inp_pos = self.flow_pos.get_mean(theta_pos, t_expanded)
        inp_ori = self.flow_ori.get_rotation(theta_ori)
        inp_ang_raw = self.flow_ang.get_angle(theta_ang_full)
        
        # CRITICAL FIX: Mask CDR sidechain angles during training to prevent data leakage
        # Sidechain torsion angles reveal amino acid identity; model should learn backbone→抯equence only
        mask_gen_ang = mask_gen.unsqueeze(-1)  # (N, L, 1)
        inp_ang = torch.where(mask_gen_ang, torch.zeros_like(inp_ang_raw), inp_ang_raw)

        # --- Recycling Pass ---
        # Training uses train_recycles (default 1); inference uses sample_opt.num_recycles.
        # Set train_recycles >= 2 to train feedback embeddings.
        train_recycles = self.loss_weight.get('train_recycles', 1)
        prev_plddt, prev_iptm, prev_pae = None, None, None
        prev_seq_emb = None  # V13

        for _ in range(train_recycles):
            # WAY4 V6/V7: epitope disorder condition for CDR generation.
            # V7 P0-fix-B: prefer per-residue profile (N, L), fall back to scalar (N, 1).
            epi_disorder = batch.get('epitope_disorder_profile', None)
            if epi_disorder is None:
                epi_disorder = batch.get('epitope_disorder', None)
            pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder, pred_contact, pred_contrastive = self.receiver(
                inp_seq, inp_pos, inp_ori, inp_ang, t, pair_feat, mask_res,
                backbone_pos=backbone_pos, prev_conf=prev_plddt, prev_iptm=prev_iptm, prev_pae=prev_pae,
                prev_seq_emb=prev_seq_emb, mask_gen=mask_gen,
                mask_antigen=batch.get('mask_antigen', None),
                epitope_disorder=epi_disorder
            )
            # Feed confidence + sequence embedding into next recycle
            prev_plddt, prev_iptm, prev_pae = pred_plddt, pred_iptm, pred_pae
            prev_seq_emb = self.receiver.v12_seq_emb(F.softmax(pred_seq.detach(), dim=-1))
        pred_contact_pair = getattr(self.receiver, 'last_pair_contact_logits', None)
        # --- End Recycling ---
        
        # 4. Training Losses
        losses = {}
        
        # BFN weighting
        weights = torch.full((N, 1), self.beta, device=x_seq.device)

        # === Sequence Loss ===
        # Run CE in fp32 (autocast disabled) for numerical stability. Under
        # AMP fp16 the receiver's sequence logits can overflow in the softmax
        # (inf - inf), producing NaN seq loss on a subset of batches — V15 was
        # skipping ~1/4 of micro-batches to NaN. fp32 CE is cheap relative to
        # the encoder and eliminates the spurious NaNs.
        x_seq_target[x_seq_target >= 20] = -100
        with torch.autocast(device_type=x_seq.device.type, enabled=False):
            loss_seq = sequence_cross_entropy_20(
                pred_seq, x_seq_target,
                label_smoothing=self.loss_weight.get('label_smoothing', 0.0))
        losses['seq'] = (loss_seq * mask_gen * weights).sum() / (mask_gen.sum() + 1e-8)

        # Directly teach profile specificity. A mismatched antigen profile must
        # assign lower likelihood to the native CDR than the factual profile.
        factual_nll = (loss_seq * mask_gen).sum(dim=1) / mask_gen.sum(dim=1).clamp(min=1)

        mismatch_w = self.loss_weight.get('antigen_mismatch_rank', 0.0)
        mismatch_aa = batch.get('antigen_mismatch_aa')
        mismatch_pair_feat = batch.get('antigen_mismatch_pair_feat')
        if mismatch_w > 0 and mismatch_aa is not None and mismatch_pair_feat is not None:
            mismatch_valid = batch.get(
                'antigen_mismatch_valid',
                torch.ones(N, dtype=torch.bool, device=x_seq.device),
            ).bool() & mask_gen.any(dim=1) & batch['mask_antigen'].any(dim=1)
            if mismatch_valid.any():
                factual_onehot = F.one_hot(
                    x_seq.clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                ).to(theta_seq.dtype)
                mismatch_onehot = F.one_hot(
                    mismatch_aa.clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                ).to(theta_seq.dtype)
                alpha = self.flow_seq._alpha(t_expanded).unsqueeze(-1)
                antigen_mask = batch['mask_antigen'].unsqueeze(-1)
                mismatch_theta = torch.where(
                    antigen_mask,
                    theta_seq + alpha * (mismatch_onehot - factual_onehot),
                    theta_seq,
                )
                mismatch_inp_seq = self.flow_seq.probabilities(mismatch_theta)
                mismatch_output = self.receiver(
                    mismatch_inp_seq, inp_pos, inp_ori, inp_ang, t,
                    mismatch_pair_feat, mask_res,
                    backbone_pos=backbone_pos, prev_conf=None, prev_iptm=None,
                    prev_pae=None, prev_seq_emb=None, mask_gen=mask_gen,
                    mask_antigen=batch['mask_antigen'], epitope_disorder=None,
                )
                mismatch_ce = sequence_cross_entropy_20(
                    mismatch_output[0], x_seq_target,
                    label_smoothing=self.loss_weight.get('label_smoothing', 0.0),
                )
                mismatch_nll = ((mismatch_ce * mask_gen).sum(dim=1)
                                / mask_gen.sum(dim=1).clamp(min=1))
                gap = factual_nll - mismatch_nll
                margin = float(self.loss_weight.get(
                    'antigen_mismatch_rank_margin', 0.05))
                losses['antigen_mismatch_rank'] = (
                    margin + gap[mismatch_valid]).clamp(min=0).mean()
                losses['antigen_mismatch_gap'] = gap[mismatch_valid].mean().detach()
                if batch.get('return_per_sample_metrics', False):
                    losses['factual_nll_per_sample'] = factual_nll.detach()
                    losses['antigen_mismatch_gap_per_sample'] = gap.detach()
                    losses['antigen_mismatch_valid_per_sample'] = mismatch_valid.detach()

        for loss_name, profile_key in (
                ('disorder_position_rank', 'epitope_disorder_shuffled_profile'),
                ('disorder_mismatch_rank', 'epitope_disorder_mismatched_profile')):
            rank_w = self.loss_weight.get(loss_name, 0.0)
            negative_profile = batch.get(profile_key)
            if rank_w <= 0 or negative_profile is None or not mask_gen.any():
                continue
            negative_output = self.receiver(
                inp_seq, inp_pos, inp_ori, inp_ang, t, pair_feat, mask_res,
                backbone_pos=backbone_pos, prev_conf=None, prev_iptm=None,
                prev_pae=None, prev_seq_emb=None, mask_gen=mask_gen,
                mask_antigen=batch.get('mask_antigen'),
                epitope_disorder=negative_profile,
            )
            negative_ce = sequence_cross_entropy_20(
                negative_output[0], x_seq_target,
                label_smoothing=self.loss_weight.get('label_smoothing', 0.0))
            negative_nll = ((negative_ce * mask_gen).sum(dim=1)
                            / mask_gen.sum(dim=1).clamp(min=1))
            margin = float(self.loss_weight.get(f'{loss_name}_margin', 0.05))
            gap = factual_nll - negative_nll
            losses[loss_name] = rank_w * (
                margin + gap).clamp(min=0).mean()
            losses[f'{loss_name}_gap'] = gap.mean().detach()

        # === Direction F: Contact prediction auxiliary loss ===
        # BCE: does each CDR residue contact ANY antigen residue (<8Å Cβ-Cβ)?
        # Gives antigen path direct supervised signal, bypassing BFN diffusion noise.
        if 'contact' in self.loss_weight and self.loss_weight.get('contact', 0) > 0:
            mask_antigen = batch.get('mask_antigen', None)
            if mask_antigen is not None and mask_antigen.any() and mask_gen.any():
                N_batch, L_batch = mask_gen.shape
                # Cβ positions (fallback to CA for GLY)
                cb = batch['pos_heavyatom'][:, :, 4]  # (N, L, 3)
                cb_mask = batch['mask_heavyatom'][:, :, 4]  # (N, L)
                ca = batch['pos_heavyatom'][:, :, 1]  # (N, L, 3)
                pos = torch.where(cb_mask.unsqueeze(-1), cb, ca)  # (N, L, 3)

                # Contact label: per CDR residue, 1 if Cβ < 8Å to any antigen residue
                contact_label = torch.zeros(N_batch, L_batch, device=pred_contact.device)
                for b in range(N_batch):
                    cdr_idx = mask_gen[b].bool()
                    ag_idx = mask_antigen[b].bool()
                    if cdr_idx.any() and ag_idx.any():
                        dists = torch.cdist(
                            contact_pos[b][cdr_idx], contact_pos[b][ag_idx])
                        contact_label[b][cdr_idx] = (dists.min(dim=1).values < 8.0).float()

                # BCE loss only on CDR residues
                loss_contact = F.binary_cross_entropy_with_logits(
                    pred_contact, contact_label, reduction='none')
                n_cdr = mask_gen.sum().clamp(min=1)
                losses['contact'] = (loss_contact * mask_gen).sum() / n_cdr

                pair_w = self.loss_weight.get('contact_pair', 0.0)
                if pair_w > 0 and pred_contact_pair is not None:
                    pair_mask = mask_gen.unsqueeze(-1) & mask_antigen.unsqueeze(1)
                    pair_label = torch.zeros_like(pred_contact_pair)
                    for b in range(N_batch):
                        if pair_mask[b].any():
                            pair_label[b] = (
                                torch.cdist(contact_pos[b], contact_pos[b]) < 8.0).float()
                    pair_loss = F.binary_cross_entropy_with_logits(
                        pred_contact_pair, pair_label, reduction='none')
                    losses['contact_pair'] = pair_w * (
                        pair_loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)

        # Direction H: contrastive CDR-antigen matching loss.
        # BCE: real CDR → 1, shuffled CDR → 0. Forces encoder to learn
        # antigen-compatible features that benefit the sequence decoder.
        if contrastive_w > 0 and contrastive_valid.any():
            loss_contrastive = F.binary_cross_entropy_with_logits(
                pred_contrastive[contrastive_valid],
                contrastive_target[contrastive_valid], reduction='mean')
            losses['contrastive'] = loss_contrastive

        grouped_contrastive_w = self.loss_weight.get('grouped_contrastive', 0)
        grouped_fields = (
            'contrastive_group_id', 'contrastive_rank', 'contrastive_weight')
        if grouped_contrastive_w > 0 and all(key in batch for key in grouped_fields):
            antigen_valid = batch['mask_antigen'].any(dim=1)
            losses['grouped_contrastive'] = grouped_contrastive_margin(
                pred_contrastive,
                batch['contrastive_group_id'],
                batch['contrastive_rank'],
                batch['contrastive_weight'],
                valid_samples=mask_gen.any(dim=1) & antigen_valid,
                margin=float(self.loss_weight.get('grouped_contrastive_margin', 0.5)),
                min_rank_gap=float(self.loss_weight.get(
                    'grouped_contrastive_min_rank_gap', 1e-6)),
            )

        # === Optional Structure Losses (only compute if in config) ===
        if any(k in self.loss_weight for k in ['dist', 'fape', 'ang']):
            pred_R = repr_6d_to_rotation_matrix(pred_ori_6d)
            pred_pos_unnorm = self._unnormalize_position(pred_pos)
            x_pos_unnorm = self._unnormalize_position(x_pos)
            
            if 'dist' in self.loss_weight and self.loss_weight['dist'] > 0:
                dist_pred = torch.cdist(pred_pos_unnorm, pred_pos_unnorm)
                dist_true = torch.cdist(x_pos_unnorm, x_pos_unnorm)
                mask_row = mask_gen.unsqueeze(-1).expand(N, mask_gen.size(1), mask_gen.size(1))
                mask_col = mask_gen.unsqueeze(1).expand(N, mask_gen.size(1), mask_gen.size(1))
                mask_pair = (mask_row | mask_col) & mask_res.unsqueeze(-1) & mask_res.unsqueeze(1)
                loss_dist = F.mse_loss(dist_pred, dist_true, reduction='none')
                losses['dist'] = (loss_dist * mask_pair).sum() / (mask_pair.sum() + 1e-8)
            
            if 'fape' in self.loss_weight and self.loss_weight['fape'] > 0:
                losses['fape'] = compute_fape(pred_R, pred_pos_unnorm, pred_pos_unnorm, x_ori, x_pos_unnorm, x_pos_unnorm, mask_gen, mask_gen)
            
            if 'ang' in self.loss_weight and self.loss_weight['ang'] > 0:
                s_hat, c_hat = pred_ang_sc[..., :4], pred_ang_sc[..., 4:]
                norm = torch.sqrt(s_hat**2 + c_hat**2 + 1e-8)
                s_hat, c_hat = s_hat / norm, c_hat / norm
                cos_diff = c_hat * torch.cos(x_ang) + s_hat * torch.sin(x_ang)
                loss_ang = 2 * (1 - cos_diff).sum(dim=-1)
                losses['ang'] = (loss_ang * mask_gen * weights).sum() / (mask_gen.sum() + 1e-8)

        # === Auxiliary Sidechain Angle Loss (for CDR regions) ===
        # This helps encoder learn geometric features from backbone
        # Input sidechain angles are masked (zero) for CDR, but we can still predict them
        if 'ang_aux' in self.loss_weight and self.loss_weight['ang_aux'] > 0:
            s_hat, c_hat = pred_ang_sc[..., :4], pred_ang_sc[..., 4:]
            norm = torch.sqrt(s_hat**2 + c_hat**2 + 1e-8)
            s_hat, c_hat = s_hat / norm, c_hat / norm
            cos_diff = c_hat * torch.cos(x_ang) + s_hat * torch.sin(x_ang)
            loss_ang_aux = 2 * (1 - cos_diff).sum(dim=-1)
            # Only compute on CDR (mask_gen) region
            losses['ang_aux'] = (loss_ang_aux * mask_gen).sum() / (mask_gen.sum() + 1e-8)

        # === Confidence Losses (pLDDT / ipTM / PAE) ===
        # V10: Huber loss for robustness against outliers (especially IDP regions).
        # Per-residue confidence weighting for pLDDT.
        huber_beta = self.loss_weight.get('huber_beta', 0.1)

        if 'af2_plddt' in batch and 'plddt' in self.loss_weight and self.loss_weight['plddt'] > 0:
            af2_plddt = batch['af2_plddt']
            pred_len = pred_plddt.shape[1]
            if af2_plddt.shape[1] < pred_len:
                af2_plddt = F.pad(af2_plddt, (0, pred_len - af2_plddt.shape[1]))
            else:
                af2_plddt = af2_plddt[:, :pred_len]
            mask_plddt = mask_res[:, :pred_len]
            # Per-residue confidence weight: higher AF2 pLDDT -> more reliable target
            conf_weight = 0.5 + 0.5 * af2_plddt  # [0.5, 1.0]
            huber = nn.SmoothL1Loss(reduction='none', beta=huber_beta)
            loss_plddt = huber(pred_plddt, af2_plddt)
            losses['plddt'] = (loss_plddt * conf_weight * mask_plddt).sum() / (mask_plddt.sum() + 1e-8)

        if 'af2_iptm' in batch and 'iptm' in self.loss_weight and self.loss_weight['iptm'] > 0:
            af2_iptm = batch['af2_iptm'].float()
            huber_iptm = nn.SmoothL1Loss(beta=huber_beta)
            losses['iptm'] = huber_iptm(pred_iptm, af2_iptm)

        if 'af2_pae_matrix' in batch and 'pae' in self.loss_weight and self.loss_weight['pae'] > 0:
            af2_pae = batch['af2_pae_matrix']
            pred_len = pred_pae.shape[1]
            if af2_pae.shape[1] < pred_len or af2_pae.shape[2] < pred_len:
                af2_pae = F.pad(af2_pae, (0, pred_len - af2_pae.shape[2], 0, pred_len - af2_pae.shape[1]))
            else:
                af2_pae = af2_pae[:, :pred_len, :pred_len]
            mask_row = mask_res[:, :pred_len].unsqueeze(-1)
            mask_col = mask_res[:, :pred_len].unsqueeze(1)
            mask_pair = mask_row & mask_col
            pae_target = af2_pae / 31.0
            huber_pae = nn.SmoothL1Loss(reduction='none', beta=huber_beta)
            loss_pae = huber_pae(pred_pae, pae_target)
            losses['pae'] = (loss_pae * mask_pair).sum() / (mask_pair.sum() + 1e-8)

        # === Disorder Loss (class-balanced BCE on confidence-weighted labels) ===
        if 'disorder_label' in batch and 'disorder' in self.loss_weight and self.loss_weight['disorder'] > 0:
            disorder_target = batch['disorder_label'].float()  # (N, L) continuous [0, 1]
            if disorder_target.dim() == 1:
                disorder_target = disorder_target.unsqueeze(0).expand(N, -1)
            pred_len = pred_disorder.shape[1]
            if disorder_target.shape[1] < pred_len:
                disorder_target = F.pad(disorder_target, (0, pred_len - disorder_target.shape[1]))
            else:
                disorder_target = disorder_target[:, :pred_len]
            mask_disorder = mask_res[:, :pred_len]
            if 'disorder_supervision_mask' in batch:
                supervision_mask = batch['disorder_supervision_mask'].bool()
                supervision_mask = supervision_mask[:, :pred_len]
                mask_disorder = mask_disorder & supervision_mask
            confidence = batch.get('disorder_confidence')
            if confidence is None:
                confidence = torch.ones_like(disorder_target)
            confidence = confidence.float()[:, :pred_len].clamp(0, 1)
            evidence_weight = confidence * mask_disorder
            per_residue_loss = F.binary_cross_entropy_with_logits(
                pred_disorder.float(), disorder_target, reduction='none')
            positive_mass = (evidence_weight * disorder_target).sum()
            negative_mass = (evidence_weight * (1.0 - disorder_target)).sum()
            if positive_mass > 0 and negative_mass > 0:
                class_weight = (
                    disorder_target / (2.0 * positive_mass)
                    + (1.0 - disorder_target) / (2.0 * negative_mass)
                )
                losses['disorder'] = (per_residue_loss * evidence_weight * class_weight).sum()
            else:
                losses['disorder'] = (
                    per_residue_loss * evidence_weight).sum() / (evidence_weight.sum() + 1e-8)

        # Rank-only supervision is permitted when calibration supports relative
        # within-protein flexibility but not absolute RMSF magnitudes.
        if ('disorder_label' in batch
                and self.loss_weight.get('disorder_rank', 0) > 0):
            disorder_target = batch['disorder_label'].float()
            if disorder_target.dim() == 1:
                disorder_target = disorder_target.unsqueeze(0).expand(N, -1)
            pred_len = pred_disorder.shape[1]
            if disorder_target.shape[1] < pred_len:
                disorder_target = F.pad(
                    disorder_target, (0, pred_len - disorder_target.shape[1]),
                    value=float('nan'))
            else:
                disorder_target = disorder_target[:, :pred_len]
            rank_mask = mask_res[:, :pred_len]
            if 'disorder_supervision_mask' in batch:
                rank_mask = rank_mask & batch['disorder_supervision_mask'].bool()[:, :pred_len]
            losses['disorder_rank'] = _within_protein_disorder_ranking(
                pred_disorder,
                disorder_target,
                rank_mask,
                temperature=float(self.loss_weight.get(
                    'disorder_rank_temperature', 1.0)),
                min_target_gap=float(self.loss_weight.get(
                    'disorder_rank_min_gap', 0.05)),
            )

        # === V12 Self-Supervised + V14 Grouped Confidence Losses ===
        # V14 fix: previous conf_variance detached pred_iptm, providing ZERO
        # gradient (cosmetic loss) — a root cause of the stuck 9.47x OC. The
        # losses are now: (a) gradient-bearing, (b) restricted to *within-scaffold*
        # design-variant pairs so they teach design-specificity, and (c) anchored
        # to real AF2 labels via a grouped margin-ranking term.
        scaffold_id = batch.get('scaffold_id', None)  # (N,) long tensor, or None
        group_mask = _within_group_pair_mask(scaffold_id, N, device=x_seq.device)  # (N,N) bool

        # 1. Variance loss: maximise WITHIN-group std of predicted ipTM (keeps grad).
        #    Same-scaffold designs MUST receive different confidence scores.
        conf_var_w = self.loss_weight.get('conf_variance', 0.0)
        if conf_var_w > 0 and 'iptm' in losses and pred_iptm.numel() > 1:
            if group_mask is not None:
                within_std = _grouped_std(pred_iptm, scaffold_id)  # mean of per-group stds
            else:
                within_std = pred_iptm.std()  # legacy: whole-batch std
            losses['conf_variance'] = conf_var_w * (-within_std.clamp(max=0.3))

        # 1b. Anti-collapse hinge (V14-unfreeze): direct counter to the constant-
        #     output failure. Only fires when within-group std drops below eps,
        #     pushing predictions apart. Unlike conf_variance (smooth linear
        #     reward, easily outweighed by the iptm MSE), this is a hard floor
        #     with a steep gradient in the collapse regime (std→0).
        anticollapse_w = self.loss_weight.get('conf_anticollapse', 0.0)
        if anticollapse_w > 0 and 'iptm' in losses and pred_iptm.numel() > 1:
            if group_mask is not None:
                std_val = _grouped_std(pred_iptm, scaffold_id)
            else:
                std_val = pred_iptm.std()
            eps_c = 0.05
            # ReLU: penalty = (eps - std)+ / eps, in [0, 1]. Max when std=0.
            collar = (eps_c - std_val).clamp(min=0.0) / eps_c
            losses['conf_anticollapse'] = anticollapse_w * collar

        # 2. Ranking loss: within a scaffold group, lower PPL → higher confidence.
        conf_rank_w = self.loss_weight.get('conf_ranking', 0.0)
        if conf_rank_w > 0 and 'iptm' in losses and pred_iptm.numel() > 1:
            # Compute PPL on-the-fly from sequence logits.
            with torch.no_grad():
                logp = torch.log_softmax(pred_seq.detach(), dim=-1)
                seq_target_safe = x_seq_target.clamp(0, self.num_classes - 1)
                nll_per_pos = -(logp.gather(-1, seq_target_safe.unsqueeze(-1)).squeeze(-1))
                real_pos = (x_seq_target >= 0).float()
                valid_mask = mask_res.bool() & (~mask_gen.bool()) & (real_pos > 0.5)
                ppl = torch.zeros(N, device=pred_seq.device)
                for b in range(N):
                    b_valid = valid_mask[b]
                    if b_valid.sum() > 1:
                        ppl[b] = torch.exp(nll_per_pos[b][b_valid].mean())
                    else:
                        ppl[b] = 1.0  # degenerate case
            ppl_diff = ppl.unsqueeze(0) - ppl.unsqueeze(1)
            iptm_diff = pred_iptm.unsqueeze(0) - pred_iptm.unsqueeze(1)
            target = (ppl_diff < -0.5).float()
            pred = torch.sigmoid(iptm_diff * 10.0)
            pair_mask = (ppl_diff.abs() > 1.0).float().detach()
            if group_mask is not None:
                pair_mask = pair_mask * group_mask.float()  # restrict to within-scaffold
            n_pairs = pair_mask.sum()
            if n_pairs > 0:
                bce = -(target * torch.log(pred + 1e-8) + (1 - target) * torch.log(1 - pred + 1e-8))
                losses['conf_ranking'] = conf_rank_w * (bce * pair_mask).sum() / n_pairs

        # 3. Grouped margin-ranking loss (V14, CORE): within a scaffold group,
        #    designs with higher REAL af2_iptm must get higher predicted ipTM.
        #    This is the supervisory signal that the synthetic-label dataset
        #    (build_foundation_dataset.py) could never provide.
        grouped_margin_w = self.loss_weight.get('grouped_margin', 0.0)
        if (grouped_margin_w > 0 and 'iptm' in losses and group_mask is not None
                and 'af2_iptm' in batch):
            losses['conf_grouped_margin'] = (
                grouped_margin_w * _grouped_margin_ranking(
                    pred_iptm, batch['af2_iptm'].float().view(-1), scaffold_id, margin=0.05)
            )


        # === WAY4 V6: Disorder-Aligned Entropy Loss (§P1) ===
        # For high-disorder epitopes: encourage CDR diversity (high entropy)
        # BUT with built-in anti-degen: penalize 6-mer repeats + T/V overuse.
        # For low-disorder epitopes: encourage deterministic (low entropy).
        disorder_align_w = self.loss_weight.get('disorder_align', 0.0)
        anti_degen_w = self.loss_weight.get('anti_degen', 0.0)
        if (disorder_align_w > 0 or anti_degen_w > 0) and mask_gen.any():
            # Per-CDR-residue AA distribution entropy from pred_seq logits
            cdr_mask = mask_gen.bool()
            device = x_seq.device
            if cdr_mask.any():
                cdr_logits = pred_seq[cdr_mask]  # (n_cdr, 20)
                cdr_probs = F.softmax(cdr_logits, dim=-1)
                log_cdr_probs = torch.log(cdr_probs + 1e-8)
                per_res_entropy = -(cdr_probs * log_cdr_probs).sum(dim=-1)  # (n_cdr,)

                # Epitope disorder: prefer per-residue, fall back to scalar
                # V7: epitope_disorder_profile has antigen disorder at antigen positions
                #     (zero at antibody positions). Take mean over antigen for loss.
                # V6: epitope_disorder is a scalar (N,1) mean over antigen.
                epi_profile = batch.get('epitope_disorder_profile', None)
                if epi_profile is not None:
                    # Per-residue profile: (N, L) — get mean over antigen residues only
                    mask_ag = batch.get('mask_antigen', None)
                    if mask_ag is not None and mask_ag.any():
                        ag_disorder = (epi_profile * mask_ag.float()).sum(dim=1) / mask_ag.sum(dim=1).clamp(min=1)
                    else:
                        # No antigen mask: use mean over non-zero positions
                        nonzero_mask = (epi_profile > 0).float()
                        ag_disorder = (epi_profile * nonzero_mask).sum(dim=1) / nonzero_mask.sum(dim=1).clamp(min=1)
                    cdr_epi_d = ag_disorder.repeat_interleave(cdr_mask.sum(dim=1))  # (n_cdr,)
                else:
                    # V6 scalar fallback
                    epi_d = batch.get('epitope_disorder', torch.zeros(N, device=x_seq.device))
                    if epi_d.dim() == 2:
                        epi_d = epi_d.squeeze(-1)  # (N,1) -> (N,)
                    cdr_epi_d = epi_d.repeat_interleave(cdr_mask.sum(dim=1))  # (n_cdr,)

                if disorder_align_w > 0:
                    scale = float(self.loss_weight.get('disorder_align_scale', 0.22))
                    low_entropy = float(self.loss_weight.get('disorder_entropy_low', 0.5))
                    high_entropy = float(self.loss_weight.get('disorder_entropy_high', 2.7))
                    disorder_scaled = (cdr_epi_d / max(scale, 1e-6)).clamp(0.0, 1.0)
                    target_entropy = low_entropy + (high_entropy - low_entropy) * disorder_scaled
                    loss_da = F.mse_loss(per_res_entropy, target_entropy)
                    losses['disorder_align'] = disorder_align_w * loss_da

                    # V8 fix1: diversity loss — penalize top-3 AA probability concentration
                    # top-3 sum low = residues spread out = real diversity (unique_aa high)
                    # top-3 sum high = few AAs dominate = fake diversity
                    diversity_w = self.loss_weight.get('diversity', 0.0)
                    if diversity_w > 0:
                        top3_probs, _ = torch.topk(cdr_probs, k=3, dim=-1)
                        top3_sum = top3_probs.sum(dim=-1)  # (n_high,) lower = more diverse
                        weights_div = disorder_scaled.clamp(min=0.05)
                        losses['diversity'] = diversity_w * (
                            top3_sum * weights_div).sum() / weights_div.sum()

                if anti_degen_w > 0:
                    # V8 fix2: penalize ANY single-AA dominance across ALL CDR residues
                    # max_prob high = one AA dominates = degenerate (not just T/V)
                    # Applies to all CDR (not just high), catches "RR degenerate" too
                    max_prob_per_res = cdr_probs.max(dim=-1).values  # (n_cdr,)
                    losses['anti_degen'] = anti_degen_w * max_prob_per_res.mean()

        losses['avg_t'] = t.mean()
        return losses

    @torch.no_grad()
    def score_fixed(self, batch, fixed_t=0.5):
        """Score the supplied sequence and backbone without running the BFN flow."""
        if not 0.0 <= float(fixed_t) <= 1.0:
            raise ValueError(f'fixed_t must be in [0, 1], got {fixed_t}')

        x_seq = batch['aa']
        N, L = x_seq.shape
        mask_res = batch['mask'].bool()
        mask_gen = batch['generate_flag'].bool() & mask_res
        if 'mask_antigen' not in batch:
            batch['mask_antigen'] = self._mask_antigen(batch, mask_gen)

        inp_seq = F.one_hot(
            x_seq.clamp(min=0, max=self.num_classes - 1),
            num_classes=self.num_classes,
        ).to(dtype=batch['pair_feat'].dtype)
        inp_pos = self._normalize_position(batch['pos_heavyatom'][:, :, 1])
        inp_ori = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1],
            batch['pos_heavyatom'][:, :, 2],
            batch['pos_heavyatom'][:, :, 0],
        )
        inp_ang = batch['torsion']
        t = torch.full((N,), float(fixed_t), device=x_seq.device, dtype=inp_pos.dtype)
        epi_disorder = batch.get('epitope_disorder_profile', None)
        if epi_disorder is None:
            epi_disorder = batch.get('epitope_disorder', None)

        result = self.receiver(
            inp_seq, inp_pos, inp_ori, inp_ang, t, batch['pair_feat'], mask_res,
            backbone_pos=batch['pos_heavyatom'][:, :, :4],
            mask_gen=mask_gen,
            mask_antigen=batch.get('mask_antigen'),
            epitope_disorder=epi_disorder,
            orientation_is_rotation=True,
        )
        (_, _, _, _, pred_plddt, pred_iptm, pred_pae, pred_disorder,
         pred_contact, pred_contrastive) = result
        pred_contact_pair = getattr(self.receiver, 'last_pair_contact_logits', None)

        residue_mask = mask_res.to(pred_plddt.dtype)
        pair_mask = (mask_res.unsqueeze(-1) & mask_res.unsqueeze(-2)).to(pred_pae.dtype)
        sample_mask = mask_res.any(dim=1).to(pred_iptm.dtype)
        return {
            'plddt': pred_plddt * residue_mask,
            'iptm': pred_iptm * sample_mask,
            'pae': pred_pae * pair_mask,
            'disorder': (pred_disorder * residue_mask
                         if pred_disorder is not None else None),
            'contact': pred_contact * residue_mask,
            'contact_pair': (pred_contact_pair * pair_mask
                             if pred_contact_pair is not None else None),
            'state_compatibility': pred_contrastive * sample_mask,
        }

    @torch.no_grad()
    def sample(self, batch, sample_opt={}):
        N, L = batch['aa'].shape
        device = batch['aa'].device
        mask_gen = batch['generate_flag'].bool()
        pair_feat = batch['pair_feat']
        mask_res = batch['mask']
        # Direction D: antigen-context mask for the cross-attention seq head.
        if 'mask_antigen' not in batch:
            batch['mask_antigen'] = self._mask_antigen(batch, mask_gen)
        deterministic = sample_opt.get('deterministic', False)  # Greedy/no noise sampling
        num_recycles = sample_opt.get('num_recycles', 1)  # Confidence feedback rounds
        # Disorder-aware sampling (balance ordered/disordered). When enabled, the
        # stochastic noise added to the CDR sequence update is scaled per-residue
        # by a pliability factor: residues the disorder head reads as disordered
        # (or an externally-supplied pliability map, e.g. epitope RMSF) get MORE
        # noise → pliable, diverse CDRs facing a disordered epitope; ordered
        # framework-facing residues get less noise. Off by default.
        disorder_guided = sample_opt.get('disorder_guided', False)
        dg_strength = sample_opt.get('disorder_guided_strength', 1.0)  # 0..1+
        # Optional external pliability map (N, L) in [0,1]; overrides model pred.
        dg_pliability = sample_opt.get('disorder_pliability', None)
        # Ground Truth Structure (always used in seq-only mode)
        x_seq = batch['aa']
        x_pos = batch['pos_heavyatom'][:, :, 1] # CA
        x_pos = self._normalize_position(x_pos)
        from disorderflow.modules.common.geometry import construct_3d_basis
        x_ori = construct_3d_basis(
            batch['pos_heavyatom'][:, :, 1],
            batch['pos_heavyatom'][:, :, 2],
            batch['pos_heavyatom'][:, :, 0],
        )
        x_ang = batch['torsion']

        # Backbone atom coordinates (N, CA, C, O) for explicit feature
        backbone_pos = batch['pos_heavyatom'][:, :, :4]  # (N, L, 4, 3)

        # Check if seq-only mode (no structure generation)
        seq_only = self.receiver.seq_only

        steps = self.num_steps
        delta_alpha = self.beta / steps
        sqrt_delta_alpha = torch.sqrt(torch.tensor(delta_alpha, device=device))

        # Feedback from previous recycle (None on first pass)
        prev_plddt, prev_iptm, prev_pae = None, None, None
        prev_seq_emb = None  # V13: per-residue sequence embedding feedback

        for recycle in range(num_recycles):
            # Initialize Priors (fresh diffusion per recycle)
            theta_seq = self.flow_seq.prior((N, L), device)

            if not seq_only:
                theta_pos = self.flow_pos.prior((N, L, 3), device)
                theta_ori = self.flow_ori.prior((N, L), device)
                theta_ang = self.flow_ang.prior((N, L), device)

            for i in range(1, steps + 1):
                t = (i - 1) / steps
                t_tensor = torch.full((N,), t, device=device)
                t_expanded = t_tensor[:, None].expand(N, L)

                # For seq-only mode: use true backbone, but MASK CDR sidechain angles
                if seq_only:
                    inp_pos = x_pos  # True backbone CA positions (expected for FixBB)
                    inp_ori = x_ori  # True backbone orientations (expected for FixBB)
                    # CRITICAL FIX: Mask CDR sidechain angles to prevent data leakage
                    # Sidechain torsion angles reveal amino acid identity (e.g., Gly has no chi angles)
                    mask_gen_ang = mask_gen.unsqueeze(-1)  # (N, L, 1)
                    inp_ang = torch.where(mask_gen_ang, torch.zeros_like(x_ang), x_ang)

                else:
                    # Sample context thetas
                    theta_pos_ctx = self.flow_pos.sample_theta(x_pos, t_expanded)
                    theta_ori_ctx = self.flow_ori.sample_theta(x_ori, t_expanded)
                    theta_ang_ctx_full = self.flow_ang.sample_theta(x_ang, t_expanded)

                    # Mix generated and context
                    mask_gen_pos = mask_gen.unsqueeze(-1)
                    theta_pos = torch.where(mask_gen_pos, theta_pos, theta_pos_ctx)
                    theta_ori = torch.where(mask_gen.view(N, L, 1, 1), theta_ori, theta_ori_ctx)
                    theta_ang = (
                        torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[0], theta_ang_ctx_full[0]),
                        torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[1], theta_ang_ctx_full[1]),
                        torch.where(mask_gen.unsqueeze(-1).unsqueeze(-1), theta_ang[2], theta_ang_ctx_full[2]),
                    )

                    inp_pos = self.flow_pos.get_mean(theta_pos, t_expanded)
                    inp_ori = self.flow_ori.get_rotation(theta_ori)
                    inp_ang = self.flow_ang.get_angle(theta_ang)

                # Sequence: always mix context
                theta_seq_ctx = self.flow_seq.sample_theta(x_seq, t_expanded)
                theta_seq = torch.where(mask_gen.unsqueeze(-1), theta_seq, theta_seq_ctx)
                inp_seq = self.flow_seq.probabilities(theta_seq)

                # Receiver forward with feedback from previous recycle
                epi_disorder = batch.get('epitope_disorder_profile', None)
                if epi_disorder is None:
                    epi_disorder = batch.get('epitope_disorder', None)
                res = self.receiver(inp_seq, inp_pos, inp_ori, inp_ang, t_tensor, pair_feat, mask_res,
                                    backbone_pos=backbone_pos,
                                    prev_conf=prev_plddt, prev_iptm=prev_iptm, prev_pae=prev_pae,
                                    prev_seq_emb=prev_seq_emb, mask_gen=mask_gen,
                                    mask_antigen=batch.get('mask_antigen', None),
                                    epitope_disorder=epi_disorder)
                pred_seq_logits, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder, pred_contact, pred_contrastive = res

                # Update sequence - deterministic mode removes noise
                pred_seq_probs = F.softmax(pred_seq_logits, dim=-1)
                if deterministic:
                    y_seq = delta_alpha * pred_seq_probs  # No noise
                else:
                    noise = torch.randn_like(pred_seq_probs)
                    if disorder_guided:
                        # Per-residue pliability in [0,1]: higher → more entropy.
                        if dg_pliability is not None:
                            plia = dg_pliability.to(device)  # (N, L) or (L,)
                            if plia.dim() == 1:
                                plia = plia.unsqueeze(0).expand(N, -1)
                        elif pred_disorder is not None:
                            plia = torch.sigmoid(pred_disorder).clamp(0.0, 1.0)  # (N, L)
                        else:
                            plia = None
                        if plia is not None:
                            # noise_scale in [1-dg_strength, 1+dg_strength] mapped from pliability
                            scale = (1.0 - dg_strength) + 2.0 * dg_strength * plia  # (N, L)
                            scale = scale.unsqueeze(-1)  # (N, L, 1) -> broadcast over classes
                            noise = noise * scale
                    y_seq = delta_alpha * pred_seq_probs + sqrt_delta_alpha * noise
                theta_seq = theta_seq + y_seq

                # Update structure (only if not seq-only)
                if not seq_only:
                    y_pos = delta_alpha * pred_pos + sqrt_delta_alpha * torch.randn_like(pred_pos)
                    theta_pos = theta_pos + y_pos

                    pred_R = repr_6d_to_rotation_matrix(pred_ori_6d)
                    y_ori = delta_alpha * pred_R + sqrt_delta_alpha * torch.randn_like(pred_R)
                    theta_ori = theta_ori + y_ori

                    pred_ang = torch.atan2(pred_ang_sc[..., :4], pred_ang_sc[..., 4:])
                    y_ang = delta_alpha * pred_ang + sqrt_delta_alpha * torch.randn_like(pred_ang)
                    y_ang_expanded = y_ang.unsqueeze(-1).expand_as(theta_ang[1])
                    theta_ang = (theta_ang[0] + delta_alpha, theta_ang[1] + y_ang_expanded, theta_ang[2] + delta_alpha)

            # Feed confidence + sequence embedding into next recycle
            prev_plddt = pred_plddt
            prev_iptm = pred_iptm
            prev_pae = pred_pae
            # V13: per-residue sequence embedding for rich feedback
            prev_seq_emb = self.receiver.v12_seq_emb(
                F.softmax(pred_seq_logits, dim=-1))
        
        # Final outputs
        if theta_seq.size(-1) > 20:
            theta_seq[..., 20:] = -1e4
        final_seq = torch.argmax(theta_seq, dim=-1)
        
        if seq_only:
            final_R = x_ori
            final_pos = self._unnormalize_position(x_pos)
        else:
            final_R = self.flow_ori.get_rotation(theta_ori)
            final_pos = self._unnormalize_position(theta_pos / self.beta)
        
        from disorderflow.modules.common.so3 import rotation_to_so3vec
        v_0 = rotation_to_so3vec(final_R)
        
        # Per-residue entropy from final step logits (sequence quality proxy)
        pred_seq_probs_final = F.softmax(pred_seq_logits[..., :20], dim=-1)
        pred_entropy = -(pred_seq_probs_final * torch.log(pred_seq_probs_final + 1e-10)).sum(dim=-1)

        result = {
            0: (v_0, final_pos, final_seq),
            'pred_logits': pred_seq_logits,  # Model's direct prediction for PPL
            'pred_entropy': pred_entropy,    # Per-residue entropy (lower = more confident)
            'plddt': pred_plddt,
            'iptm': pred_iptm,
            'pae': pred_pae,
            'contact': pred_contact,
            'state_compatibility': pred_contrastive,
        }
        if sample_opt.get('return_disorder', False) and pred_disorder is not None:
            result['disorder'] = pred_disorder
        return result

    @torch.no_grad()
    def optimize(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
