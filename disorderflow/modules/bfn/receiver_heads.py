"""Reusable confidence/discriminator head modules for BFN receiver.

Extracted from receiver.py (V18 refactor). Imported by AntibodyBFN_Receiver
to keep the receiver class focused on the forward pass orchestration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
