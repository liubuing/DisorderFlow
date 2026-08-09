import torch
import torch.nn as nn

from disorderflow.modules.common.geometry import construct_3d_basis
from disorderflow.modules.encoders.residue import ResidueEmbedding
from disorderflow.modules.encoders.pair import PairEmbedding
from disorderflow.modules.bfn.core import AntibodyBFN_Core
from disorderflow.utils.protein.constants import (
    max_num_heavyatoms,
    BBHeavyAtom,
    Fragment,
)
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


def build_antigen_sequence_mismatch(batch):
    """Swap antigen sequences across a batch while preserving target geometry."""
    if 'hard_antigen_mismatch_aa' in batch:
        mismatch = batch['hard_antigen_mismatch_aa'].to(batch['aa'].device)
        valid = batch.get('hard_antigen_mismatch_valid')
        if valid is None:
            valid = (mismatch != batch['aa']).any(dim=1)
        return mismatch, valid.to(batch['aa'].device).bool()
    aa = batch['aa']
    if aa.shape[0] < 2 or 'fragment_type' not in batch:
        return None, None

    antigen = (batch['fragment_type'] == int(Fragment.Antigen)) & batch['mask'].bool()
    mismatch = aa.clone()
    valid = torch.zeros(aa.shape[0], dtype=torch.bool, device=aa.device)
    for target in range(aa.shape[0]):
        donor = (target + 1) % aa.shape[0]
        target_idx = torch.where(antigen[target])[0]
        donor_idx = torch.where(antigen[donor])[0]
        if target_idx.numel() == 0 or donor_idx.numel() == 0:
            continue
        mapped = torch.linspace(
            0, donor_idx.numel() - 1, target_idx.numel(), device=aa.device
        ).round().long()
        donor_aa = aa[donor, donor_idx[mapped]]
        mismatch[target, target_idx] = donor_aa
        valid[target] = not torch.equal(aa[target, target_idx], donor_aa)
    return mismatch, valid


@register_model('antibody_bfn')
class AntibodyBFN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.bfn = AntibodyBFN_Core(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            cfg.diffusion.num_steps, # Reuse num_steps config
            eps_net_opt=cfg.diffusion.eps_net_opt, # Reuse eps_net_opt
            loss_weight=cfg.get('loss_weight', {}), # Pass loss weights from config
            ot_opt=cfg.get('ot_opt', {}), # Pass OT options
            beta=cfg.diffusion.get('beta', 1.0),  # BFN precision (default 1.0)
            schedule=cfg.diffusion.get('schedule', 'linear'),  # 'linear' or 'cosine'
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
            generate_flag = batch.get('generate_flag'),
        )

        return res_feat, pair_feat
    
    def forward(self, batch):
        # In BFN training, we feed the "Clean" data to the encoder to get context embeddings.
        # But wait, BFN flow handles noise.
        # The 'context' part of the protein is clean. The 'generated' part is noisy.
        # The embedding layers (pair_embed) take the whole structure.
        # If we feed clean structure to pair_embed, we cheat?
        # In Diffusion, we noise the structure BEFORE embedding? No, in DiffAb, 'encode' takes 'batch'.
        # And 'batch' contains clean data.
        # Inside 'encode', it uses 'structure_mask' to mask out generated regions for embeddings.
        # So 'pair_feat' only contains info from context.
        # This logic is preserved here.
        
        res_feat, pair_feat = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        
        # Pass pair_feat to batch for BFN core
        batch['pair_feat'] = pair_feat
        # We might pass res_feat too if needed, but BFN receiver builds its own res features from theta.
        # However, the context part of res_feat is useful.
        # The current BFN Core prototype doesn't mix context res_feat.
        # We should probably improve Receiver to take context res_feat.
        batch['res_feat'] = res_feat

        if self.cfg.get('loss_weight', {}).get('antigen_mismatch_rank', 0) > 0:
            mismatch_aa, mismatch_valid = build_antigen_sequence_mismatch(batch)
            if mismatch_aa is not None:
                mismatch_batch = dict(batch)
                mismatch_batch['aa'] = mismatch_aa
                _, mismatch_pair_feat = self.encode(
                    mismatch_batch,
                    remove_structure=self.cfg.get('train_structure', True),
                    remove_sequence=self.cfg.get('train_sequence', True),
                )
                batch['antigen_mismatch_aa'] = mismatch_aa
                batch['antigen_mismatch_pair_feat'] = mismatch_pair_feat
                batch['antigen_mismatch_valid'] = mismatch_valid

        loss_dict = self.bfn(batch)
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        res_feat, pair_feat = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        batch['pair_feat'] = pair_feat
        batch['res_feat'] = res_feat
        
        traj = self.bfn.sample(batch, sample_opt)
        return traj

    @torch.no_grad()
    def score(self, batch, fixed_t=0.5):
        res_feat, pair_feat = self.encode(
            batch,
            remove_structure=False,
            remove_sequence=False,
        )
        batch['pair_feat'] = pair_feat
        batch['res_feat'] = res_feat
        return self.bfn.score_fixed(batch, fixed_t=fixed_t)

    @torch.no_grad()
    def optimize(self, batch, *args, **kwargs):
        return self.sample(batch, *args, **kwargs)
