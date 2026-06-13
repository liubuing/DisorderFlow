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
    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}, num_classes=20, seq_only=False, head_dropout=0.1, disorder_head=False, encoder_dropout=0.0):
        super().__init__()
        self.res_feat_dim = res_feat_dim
        self.num_classes = num_classes
        self.seq_only = seq_only
        self.disorder_head = disorder_head
        
        # Embeddings
        self.seq_embed = nn.Linear(num_classes, res_feat_dim)
        self.pos_embed = nn.Linear(3, res_feat_dim) # Embed mean position? 
        # Actually GAEncoder takes 't' directly. Maybe we just use pos directly.
        
        self.angle_embed = nn.Linear(4 * 2, res_feat_dim) # Concatenate sin/cos of 4 chi angles
        
        # Backbone atom coordinates embedding (N, CA, C, O relative to CA)
        # 4 atoms 脳 3 coords = 12 features
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

        # Output Heads
        self.head_seq = nn.Linear(res_feat_dim, num_classes)
        self.head_pos = nn.Linear(res_feat_dim, 3)
        self.head_ori = nn.Linear(res_feat_dim, 6) # 6D representation for rotation
        self.head_ang = nn.Linear(res_feat_dim, 4 * 2) # sin/cos for 4 angles

        # Confidence Heads (V10: deeper residual MLPs with dropout)
        self.head_plddt = ResidualMLP(res_feat_dim, res_feat_dim, 1, num_blocks=3, dropout=head_dropout)
        self.head_iptm = ResidualMLP(res_feat_dim, res_feat_dim // 2, 1, num_blocks=2, dropout=head_dropout)
        self.head_pae = AttentionPAEHead(
            res_dim=res_feat_dim, pair_dim=pair_feat_dim,
            num_heads=4, dropout=head_dropout,
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

        # Feedback embedding: pLDDT, ipTM, and PAE back to features
        self.conf_embed = nn.Linear(1, res_feat_dim)
        self.iptm_embed = nn.Linear(1, res_feat_dim)
        self.pae_embed = nn.Linear(1, pair_feat_dim)
        

    def forward(self, theta_seq, theta_pos, theta_ori, theta_ang, t, pair_feat, mask_res,
                backbone_pos=None, prev_conf=None, prev_iptm=None, prev_pae=None, mask_gen=None):
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
        if prev_conf is not None:
            res_feat = res_feat + self.conf_embed(prev_conf.unsqueeze(-1))
        if prev_iptm is not None:
            res_feat = res_feat + self.iptm_embed(prev_iptm.view(-1, 1, 1))
        if prev_pae is not None:
            pair_feat = pair_feat + self.pae_embed(prev_pae.unsqueeze(-1))
        
        # 2. Encoder
        features = self.encoder(rot, pos, res_feat, pair_feat, mask_res)
        
        # 3. Heads
        pred_seq = self.head_seq(features)
        if pred_seq.size(-1) > 20:
            pred_seq[..., 20:] = -1e4

        # Disorder prediction (per-residue binary logits)
        if self.disorder_head:
            pred_disorder = self.head_disorder(features).squeeze(-1)  # (N, L)
        else:
            pred_disorder = None

        # Seq-only mode: skip structure predictions (pos, ori) but keep angle prediction for aux loss
        # Confidence predictions (pLDDT/ipTM/PAE) are still computed for fine-tuning
        if self.seq_only:
            pred_pos = pos  # Just return input
            pred_ori_6d = torch.zeros(N, L, 6, device=device)
            pred_ang_sc = self.head_ang(features)  # Still predict angles for auxiliary loss
            pred_plddt = torch.sigmoid(self.head_plddt(features)).squeeze(-1)

            masked_features = features * mask_res.unsqueeze(-1)

            # All-residue ipTM: pool over both context AND generated residues.
            # Previously pooled only over context (non-generated) residues, which
            # made ipTM constant across designs since context residues are identical.
            all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
            all_feat = masked_features.sum(dim=1) / all_sum
            pred_iptm = torch.sigmoid(self.head_iptm(all_feat)).squeeze(-1)

            pred_pae = torch.sigmoid(self.head_pae(features, pair_feat))
        else:
            # Full prediction
            pred_pos_local = self.head_pos(features)
            pred_pos = torch.matmul(rot, pred_pos_local.unsqueeze(-1)).squeeze(-1) + pos
            pred_ori_6d = self.head_ori(features)
            pred_ang_sc = self.head_ang(features)
            pred_plddt = torch.sigmoid(self.head_plddt(features)).squeeze(-1)

            masked_features = features * mask_res.unsqueeze(-1)
            all_sum = mask_res.sum(dim=1, keepdim=True).clamp(min=1)
            all_feat = masked_features.sum(dim=1) / all_sum
            pred_iptm = torch.sigmoid(self.head_iptm(all_feat)).squeeze(-1)

            pred_pae = torch.sigmoid(self.head_pae(features, pair_feat))
        return pred_seq, pred_pos, pred_ori_6d, pred_ang_sc, pred_plddt, pred_iptm, pred_pae, pred_disorder
