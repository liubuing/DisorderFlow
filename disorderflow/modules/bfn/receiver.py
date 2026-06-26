import torch
import torch.nn as nn
import torch.nn.functional as F
from disorderflow.modules.encoders.ga import GAEncoder
from disorderflow.modules.bfn.orientation import svd_project_so3


class ResidualMLP(nn.Module):
    """MLP with residual blocks, LayerNorm, and Dropout for confidence heads.

    Structure:
        input -> Linear(in_dim, hidden_dim) -> ReLU -> [ResBlock x N] -> Linear(hidden_dim, out_dim)

    where each ResBlock is:
        x -> LinearNorm(hidden) -> ReLU -> Dropout -> + x (residual)
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)


class AttentionPAEHead(nn.Module):
    """Lightweight attention-based PAE head with pair feature bias.

    Uses Q/K outer product to capture pairwise relationships,
    with pair_feat (geometric/evolutionary features) as attention bias.
    Output: (N, L, L) pairwise error prediction.
    """

    def __init__(self, res_dim=256, pair_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = res_dim // 2 // num_heads  # 128 // 4 = 32
        self.head_dim = head_dim
        self.q_proj = nn.Linear(res_dim, res_dim // 2)    # 256 -> 128
        self.k_proj = nn.Linear(res_dim, res_dim // 2)    # 256 -> 128
        self.pair_bias_proj = nn.Linear(pair_dim, num_heads)  # 128 -> 4
        self.output = nn.Sequential(
            nn.Linear(num_heads + pair_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features, pair_feat):
        N, L, D = features.shape
        Q = self.q_proj(features)  # (N, L, 128)
        K = self.k_proj(features)  # (N, L, 128)
        Q = Q.view(N, L, self.num_heads, self.head_dim)  # (N, L, 4, 32)
        K = K.view(N, L, self.num_heads, self.head_dim)  # (N, L, 4, 32)
        attn = torch.einsum('nihd,njhd->nijh', Q, K) / (self.head_dim ** 0.5)
        pair_bias = self.pair_bias_proj(pair_feat)  # (N, L, L, 4)
        attn = attn + pair_bias
        attn = F.relu(attn)
        combined = torch.cat([attn, pair_feat], dim=-1)  # (N, L, L, 4+128=132)
        return self.output(combined).squeeze(-1)  # (N, L, L)


class AntibodyBFN_Receiver(nn.Module):
    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}, num_classes=20, seq_only=False, head_dropout=0.1, disorder_head=False, encoder_dropout=0.0, confidence_version='v12'):
        super().__init__()
        self.res_feat_dim = res_feat_dim
        self.num_classes = num_classes
        self.seq_only = seq_only
        self.disorder_head = disorder_head
        self.confidence_version = confidence_version
        
        # Embeddings
        self.seq_embed = nn.Linear(num_classes, res_feat_dim)
        self.pos_embed = nn.Linear(3, res_feat_dim) # Embed mean position? 
        # Actually GAEncoder takes 't' directly. Maybe we just use pos directly.
        
        self.angle_embed = nn.Linear(4 * 2, res_feat_dim) # Concatenate sin/cos of 4 chi angles
        
        # Backbone atom coordinates embedding (N, CA, C, O relative to CA)
        # 4 atoms × 3 coords = 12 features
        self.backbone_embed = nn.Linear(12, res_feat_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim)
        )

        self.res_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 4, res_feat_dim), # seq + angle + time + backbone
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim)
        )

        # Shared Geometry-Aware Encoder
        # Ensure num_layers is not duplicated in encoder_opt
        opt = encoder_opt.copy()
        if 'num_layers' in opt:
            num_layers = opt.pop('num_layers')
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, ga_block_opt=opt, dropout=encoder_dropout)

        # ── Direction D: antigen→CDR cross-attention into the sequence head ──
        # probe_antigen_signal_usage showed complex-vs-fixbb CDR recovery is
        # identical (Δ +0.1pp): the antigen chain reaches the encoder via
        # pair_feat but NEVER reaches the per-residue head_seq (Linear, reads
        # only the CDR residue's own features). Fix: a cross-attention block
        # where CDR residues query antigen-residue encoder features, then gate-
        # fuse as a RESIDUAL onto features before head_seq. Residual (not concat)
        # preserves the V15-trained head_seq weights — only the new antigen_xa
        # block is fresh-init. In fixbb (no antigen) the path is a no-op.
        self.antigen_xa = nn.MultiheadAttention(res_feat_dim, num_heads=4, kdim=res_feat_dim,
                                                vdim=res_feat_dim, batch_first=True, dropout=head_dropout)
        self.antigen_xa_norm = nn.LayerNorm(res_feat_dim)
        self.antigen_xa_gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3)≈0.05: near-inert at start (V15 behavior preserved), learns up

        # Output Heads
        self.head_seq = nn.Linear(res_feat_dim, num_classes)
        self.head_pos = nn.Linear(res_feat_dim, 3)
        self.head_ori = nn.Linear(res_feat_dim, 6) # 6D representation for rotation
        self.head_ang = nn.Linear(res_feat_dim, 4 * 2) # sin/cos for 4 angles

        # ── Confidence Heads ──
        # V11 (default): backbone confidence + sequence-aware bypass
        # V12 (experimental): pure-sequence confidence, no backbone path
        if confidence_version == 'v12':
            # V12: Pure-sequence confidence (no trained checkpoint yet)
            NC = num_classes
            SEQ_H = 128
            self.v12_plddt = nn.Sequential(
                nn.Linear(NC, SEQ_H), nn.LayerNorm(SEQ_H), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(SEQ_H, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(64, 1),
            )
            self.v12_seq_emb = nn.Sequential(
                nn.Linear(NC, 64), nn.LayerNorm(64), nn.ReLU(),
            )
            self.v12_iptm = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(32, 1),
            )
            # V14-unfreeze: backbone-geometry pathway for ipTM, FRESH init.
            # The v12 ipTM head is pure-sequence and was poisoned by synthetic
            # constant labels (probe_conf_collapse.py: val pred std=1.8e-5).
            # This fresh ResidualMLP on pooled encoder features gives the head
            # real structural signal AND an escape route out of the constant
            # basin via randomly-initialized weights.
            self.v14_iptm_bb = ResidualMLP(res_feat_dim, res_feat_dim // 2, 1, num_blocks=2, dropout=head_dropout)
            self.v12_pae = nn.Sequential(
                nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
            )
            # Legacy backbone heads for V12 checkpoint compatibility
            self.head_plddt = ResidualMLP(res_feat_dim, res_feat_dim, 1, num_blocks=3, dropout=head_dropout)
            self.head_iptm = ResidualMLP(res_feat_dim, res_feat_dim // 2, 1, num_blocks=2, dropout=head_dropout)
            self.head_pae = AttentionPAEHead(res_dim=res_feat_dim, pair_dim=pair_feat_dim, num_heads=4, dropout=head_dropout)
        else:
            # V11 (default): backbone confidence + sequence-aware bypass
            # Backbone confidence heads
            self.head_plddt = ResidualMLP(res_feat_dim, res_feat_dim, 1, num_blocks=3, dropout=head_dropout)
            self.head_iptm = ResidualMLP(res_feat_dim, res_feat_dim // 2, 1, num_blocks=2, dropout=head_dropout)
            self.head_pae = AttentionPAEHead(res_dim=res_feat_dim, pair_dim=pair_feat_dim, num_heads=4, dropout=head_dropout)
            # Sequence-aware bypass (V11: fixes FixBB overconfidence)
            SEQ_CONF_IN_DIM = 2  # entropy + max_prob per residue
            self.head_plddt_seq = nn.Sequential(
                nn.Linear(SEQ_CONF_IN_DIM, 32), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(32, 1),
            )
            self.head_iptm_seq = nn.Sequential(
                nn.Linear(SEQ_CONF_IN_DIM, 16), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(16, 1),
            )
            self.head_pae_seq = nn.Sequential(
                nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(head_dropout),
                nn.Linear(32, 1),
            )

        # Disorder Classification Head (per-residue binary)
        if disorder_head:
            self.head_disorder = nn.Sequential(
                nn.Linear(res_feat_dim, res_feat_dim // 2),
                nn.LayerNorm(res_feat_dim // 2),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(res_feat_dim // 2, 1),
            )

        # Feedback embedding: pLDDT, ipTM, PAE, AND sequence embedding (V13)
        self.conf_embed = nn.Linear(1, res_feat_dim)
        self.iptm_embed = nn.Linear(1, res_feat_dim)
        self.pae_embed = nn.Linear(1, pair_feat_dim)
        # V13: per-residue sequence embedding feedback — much richer than scalar conf
        self.v13_seq_feedback = nn.Linear(64, res_feat_dim)
        # V14 Complex-mode fix: learnable gates on scalar confidence feedback
        # channels, initialised low (0.3) to stop backbone-constant confidence
        # from drowning the epitope signal during recycling. The seq_emb channel
        # (above) stays ungated.
        self.recycle_gate_conf = nn.Parameter(torch.tensor(0.3))
        self.recycle_gate_iptm = nn.Parameter(torch.tensor(0.3))
        self.recycle_gate_pae = nn.Parameter(torch.tensor(0.3))
        # B1 diagnostic: None by default so training-time Receivers
        # have the attribute (probe scripts set it to [] to enable stash).
        self._dbg_iptm_components = None

    def forward(self, theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res,
                backbone_pos=None, prev_conf=None, prev_iptm=None, prev_pae=None,
                prev_seq_emb=None, mask_gen=None, mask_antigen=None):
        N, L, _ = theta_seq.shape
        device = theta_seq.device
        
        # 1. Embeddings
        # Sequence: theta_seq is logits/accumulated evidence. Softmax to get probabilities.
        probs_seq = F.softmax(theta_seq, dim=-1) # (N, L, 20)
        emb_seq = self.seq_embed(probs_seq)      # (N, L, D)
        
        # Position: theta_pos is mean.
        # We pass this as 't' (translation) to GAEncoder.
        pos = theta_pos 
        
        # Orientation: theta_ori is F matrix. Project to R.
        # We pass this as 'R' to GAEncoder.
        rot = svd_project_so3(theta_ori)
        
        # Angle: theta_ang. Let's assume we extract expected angles.
        # For prototype, we assume theta_ang is (log_weights, means, precisions)
        # simplified: just use the means of the most likely component or weighted sum.
        # Let's use the 'get_angle' logic or simply pass the first moment if implemented.
        # Here we just take the means tensor directly if passed (simplified flow).
        # Assuming theta_ang passed here is already processed to (N, L, 4) or similar.
        # Wait, the core calls this. 
        # Let's assume theta_ang has shape (N, L, 4) representing estimated angles for embedding.
        # We embed as sin/cos.
        s_ang = torch.sin(theta_ang)
        c_ang = torch.cos(theta_ang)
        emb_ang = self.angle_embed(torch.cat([s_ang, c_ang], dim=-1)) # (N, L, D)
        
        # Time
        emb_t = self.time_embed(t.view(-1, 1, 1).expand(emb_seq.shape[:2] + (1,)))
        
        # Backbone coordinates (N, CA, C, O relative to CA)
        if backbone_pos is not None:
            # backbone_pos: (N, L, 4, 3) - positions of N, CA, C, O atoms
            ca_pos = backbone_pos[:, :, 1:2, :]  # (N, L, 1, 3) - CA position
            backbone_rel = backbone_pos - ca_pos  # (N, L, 4, 3) relative to CA
            backbone_flat = backbone_rel.reshape(N, L, -1)  # (N, L, 12)
            emb_backbone = self.backbone_embed(backbone_flat)  # (N, L, D)
        else:
            emb_backbone = torch.zeros(N, L, self.res_feat_dim, device=device)

        # Mix features
        res_feat = self.res_mixer(torch.cat([emb_seq, emb_ang, emb_t, emb_backbone], dim=-1))
        
        # Feedback recycling
        # Complex-mode fix: the scalar confidence channels (prev_conf/iptm/pae)
        # are quasi-constant across designs on the same backbone (backbone-
        # dominated), so feeding them at full strength drowns the encoder's
        # epitope attention signal and makes Complex output ~= FixBB. We gate
        # these scalar channels with learnable gates initialised LOW (0.3), so
        # the design-specific prev_seq_emb channel (ungated, V13) dominates the
        # recycle feedback. The gates are learned; a frozen-backbone Phase A
        # leaves them at init, which already reduces the drowning effect.
        if prev_conf is not None:
            res_feat = res_feat + self.recycle_gate_conf * self.conf_embed(prev_conf.unsqueeze(-1))
        if prev_iptm is not None:
            res_feat = res_feat + self.recycle_gate_iptm * self.iptm_embed(prev_iptm.view(-1, 1, 1))
        if prev_pae is not None:
            pair_feat = pair_feat + self.recycle_gate_pae * self.pae_embed(prev_pae.unsqueeze(-1))
        if prev_seq_emb is not None:
            res_feat = res_feat + self.v13_seq_feedback(prev_seq_emb)

        # 2. Encoder
        features = self.encoder(rot, pos, res_feat, pair_feat, mask_res)

        # ── Sequence confidence features: entropy/max_prob for seq-aware bypass ──
        log_probs = torch.log(probs_seq + 1e-8)
        entropy = -(probs_seq * log_probs).sum(dim=-1)  # (N, L), 0=peaked, ~3.0=uniform
        max_prob = probs_seq.max(dim=-1).values         # (N, L), 1.0=certain, 0.05=blind

        # Direction D: antigen→CDR cross-attention. CDR residues query the
        # antigen residues' encoder features; the attended antigen context is
        # gated and fused as a RESIDUAL onto the CDR features before head_seq.
        # In fixbb (no antigen) mask_antigen is empty → no-op, so the path
        # stays inert (matches the seq_only baseline). With an antigen present,
        # this is the ONLY place the antigen explicitly conditions CDR decoding.
        if mask_antigen is not None and mask_antigen.any() and mask_gen is not None:
            ag_mask = mask_antigen.bool()          # (N, L)
            key_padding_mask = ~ag_mask             # (N, L) True where NOT antigen (ignore)
            xa_out, _ = self.antigen_xa(features, features, features,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False)
            # only CDR residues get the antigen update (framework stays as-is)
            xa_out = torch.where(mask_gen.unsqueeze(-1), xa_out, torch.zeros_like(xa_out))
            gate = torch.sigmoid(self.antigen_xa_gate)
            features = self.antigen_xa_norm(features + gate * xa_out)

        # 3. Heads
        pred_seq = self.head_seq(features)
        if pred_seq.size(-1) > 20:
            pred_seq[..., 20:] = -1e4

        # Disorder prediction (per-residue binary logits)
        if self.disorder_head:
            pred_disorder = self.head_disorder(features).squeeze(-1)
        else:
            pred_disorder = None

        if self.seq_only:
            pred_pos = pos
            pred_ori_6d = torch.zeros(N, L, 6, device=device)
            pred_ang_sc = self.head_ang(features)

            if self.confidence_version == 'v12':
                plddt_seq = self.v12_plddt(probs_seq)
                pred_plddt = torch.sigmoid(plddt_seq).squeeze(-1)
                seq_emb = self.v12_seq_emb(probs_seq)
                all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
                pooled_seq = (seq_emb * mask_res.unsqueeze(-1)).sum(dim=1) / all_sum
                iptm_seq = self.v12_iptm(pooled_seq)
                # V14-unfreeze: add fresh backbone-geometry pathway (see __init__).
                masked_features = features * mask_res.unsqueeze(-1)
                all_feat = masked_features.sum(dim=1) / all_sum
                iptm_bb = self.v14_iptm_bb(all_feat)
                if self._dbg_iptm_components is not None:
                    self._dbg_iptm_components = (iptm_seq.detach().cpu(), iptm_bb.detach().cpu())
                # B1 fix: per-pathway sigmoid + blend, NOT additive logits.
                # The old sigmoid(iptm_seq + iptm_bb) let v14_iptm_bb's -5.27
                # negative bias push the combined logit into deep sigmoid
                # saturation (-4.5), flattening pred_iptm to a near-constant
                # across CDR variants (probe: design std=1e-6, rho=0.30).
                # Blending two sigmoids keeps each pathway's dynamic range
                # independent, so neither bias can drown the other.
                pred_iptm = (0.5 * torch.sigmoid(iptm_seq) + 0.5 * torch.sigmoid(iptm_bb)).squeeze(-1)
                pae_pair_feat = torch.stack([
                    entropy.unsqueeze(-1) * entropy.unsqueeze(-2),
                    (entropy.unsqueeze(-1) - entropy.unsqueeze(-2)).abs(),
                    max_prob.unsqueeze(-1) * max_prob.unsqueeze(-2),
                ], dim=-1)
                N_p, L_p = pae_pair_feat.shape[0], pae_pair_feat.shape[1]
                pae_seq = self.v12_pae(pae_pair_feat.reshape(-1, 3)).reshape(N_p, L_p, L_p)
                pred_pae = torch.sigmoid(pae_seq)
            else:
                # V11: backbone confidence + sequence-aware bypass
                seq_conf = torch.stack([entropy, max_prob], dim=-1)
                plddt_backbone = self.head_plddt(features)
                plddt_seq = self.head_plddt_seq(seq_conf)
                pred_plddt = torch.sigmoid(plddt_backbone + plddt_seq).squeeze(-1)
                masked_features = features * mask_res.unsqueeze(-1)
                all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
                all_feat = masked_features.sum(dim=1) / all_sum
                iptm_backbone = self.head_iptm(all_feat)
                all_seq_conf = (seq_conf * mask_res.unsqueeze(-1)).sum(dim=1) / all_sum
                iptm_seq = self.head_iptm_seq(all_seq_conf)
                pred_iptm = torch.sigmoid(iptm_backbone + iptm_seq).squeeze(-1)
                pae_backbone = self.head_pae(features, pair_feat)
                ent_i = entropy.unsqueeze(-1)
                ent_j = entropy.unsqueeze(-2)
                pae_seq_feat = torch.cat([(ent_i - ent_j).abs(), ent_i * ent_j], dim=-1)
                N_p, L_p = pae_seq_feat.shape[0], pae_seq_feat.shape[1]
                pae_seq = self.head_pae_seq(pae_seq_feat.reshape(-1, 2)).reshape(N_p, L_p, L_p)
                pred_pae = torch.sigmoid(pae_backbone + pae_seq)
        else:
            # Full prediction
            pred_pos_local = self.head_pos(features)
            pred_pos = torch.matmul(rot, pred_pos_local.unsqueeze(-1)).squeeze(-1) + pos
            pred_ori_6d = self.head_ori(features)
            pred_ang_sc = self.head_ang(features)

            if self.confidence_version == 'v12':
                plddt_seq = self.v12_plddt(probs_seq)
                pred_plddt = torch.sigmoid(plddt_seq).squeeze(-1)
                seq_emb = self.v12_seq_emb(probs_seq)
                all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
                pooled_seq = (seq_emb * mask_res.unsqueeze(-1)).sum(dim=1) / all_sum
                iptm_seq = self.v12_iptm(pooled_seq)
                # V14-unfreeze: add fresh backbone-geometry pathway (see __init__).
                masked_features = features * mask_res.unsqueeze(-1)
                all_feat = masked_features.sum(dim=1) / all_sum
                iptm_bb = self.v14_iptm_bb(all_feat)
                if self._dbg_iptm_components is not None:
                    self._dbg_iptm_components = (iptm_seq.detach().cpu(), iptm_bb.detach().cpu())
                # B1 fix: per-pathway sigmoid + blend, NOT additive logits.
                # The old sigmoid(iptm_seq + iptm_bb) let v14_iptm_bb's -5.27
                # negative bias push the combined logit into deep sigmoid
                # saturation (-4.5), flattening pred_iptm to a near-constant
                # across CDR variants (probe: design std=1e-6, rho=0.30).
                # Blending two sigmoids keeps each pathway's dynamic range
                # independent, so neither bias can drown the other.
                pred_iptm = (0.5 * torch.sigmoid(iptm_seq) + 0.5 * torch.sigmoid(iptm_bb)).squeeze(-1)
                pae_pair_feat = torch.stack([
                    entropy.unsqueeze(-1) * entropy.unsqueeze(-2),
                    (entropy.unsqueeze(-1) - entropy.unsqueeze(-2)).abs(),
                    max_prob.unsqueeze(-1) * max_prob.unsqueeze(-2),
                ], dim=-1)
                N_full, L_full = pae_pair_feat.shape[0], pae_pair_feat.shape[1]
                pae_seq = self.v12_pae(pae_pair_feat.reshape(-1, 3)).reshape(N_full, L_full, L_full)
                pred_pae = torch.sigmoid(pae_seq)
            else:
                # V11: backbone confidence + sequence-aware bypass
                seq_conf = torch.stack([entropy, max_prob], dim=-1)
                plddt_backbone = self.head_plddt(features)
                plddt_seq = self.head_plddt_seq(seq_conf)
                pred_plddt = torch.sigmoid(plddt_backbone + plddt_seq).squeeze(-1)
                masked_features = features * mask_res.unsqueeze(-1)
                all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
                all_feat = masked_features.sum(dim=1) / all_sum
                iptm_backbone = self.head_iptm(all_feat)
                all_seq_conf = (seq_conf * mask_res.unsqueeze(-1)).sum(dim=1) / all_sum
                iptm_seq = self.head_iptm_seq(all_seq_conf)
                pred_iptm = torch.sigmoid(iptm_backbone + iptm_seq).squeeze(-1)
                pae_backbone = self.head_pae(features, pair_feat)
                ent_i = entropy.unsqueeze(-1)
                ent_j = entropy.unsqueeze(-2)
                pae_seq_feat = torch.cat([(ent_i - ent_j).abs(), ent_i * ent_j], dim=-1)
                N_full, L_full = pae_seq_feat.shape[0], pae_seq_feat.shape[1]
                pae_seq = self.head_pae_seq(pae_seq_feat.reshape(-1, 2)).reshape(N_full, L_full, L_full)
                pred_pae = torch.sigmoid(pae_backbone + pae_seq)

        return pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder
