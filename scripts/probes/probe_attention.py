import os
#!/usr/bin/env python
"""Probe BFN encoder attention: does scaffold attend to epitope?

Hooks into GAEncoder blocks during forward pass, extracts attention weights,
and analyzes what fraction of scaffold CDR attention goes to epitope residues.

Usage:
  python probe_attention.py
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import yaml, time, re
import torch
import numpy as np

CDR_SPEC = 'B:26-33,51-58,97-113'
V11_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'


def parse_cdr_indices(region_spec):
    """Parse region spec into flat list of 0-based residue indices."""
    indices = []
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', region_spec):
        for seg in spec.split(','):
            seg = seg.strip()
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                indices.append(int(seg))
    return sorted(set(indices))


def run_attention_probe():
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.transforms import get_transform
    from antibody_epitope_complex import position_epitope

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Setup checkpoint ──
    cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
    with open(cfg_path) as f:
        app_cfg = yaml.safe_load(f)
    orig_ckpt = app_cfg['models']['bfn']['checkpoint']
    app_cfg['models']['bfn']['checkpoint'] = V11_CKPT
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)

    try:
        from bfn_loader import load_bfn
        import bfn_loader
        bfn_loader._bfn_model = None
        bfn_loader._bfn_config = None
        model, config = load_bfn(device)
        model.eval()

        # ── Build Complex mode structure ──
        complex_info = position_epitope(
            'data/misfolding_targets/5IMK.pdb',
            'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
            scaffold_chain='B', epitope_chain='A', distance=6.0,
        )
        pdb_path = complex_info['pdb_path']
        print(f'Complex PDB: {pdb_path}')

        # ── Preprocess structure ──
        from disorderflow.utils.misc import seed_all
        seed_all(42)

        # Parse region spec
        design_chains = ['B']
        context_chains = ['A']  # epitope visible
        all_chains = ['B', 'A']

        cdr_indices = parse_cdr_indices(CDR_SPEC)  # 1-based from PDB
        regions = {'B': cdr_indices}

        structure = preprocess_protein_structure(pdb_path, chain_ids=all_chains)
        if structure is None:
            print('ERROR: cannot parse structure')
            return

        # Get chain lengths to map residue indices
        # structure should have pos_heavyatom with shape (L, 4, 3)
        L_total = structure['pos_heavyatom'].shape[0]
        print(f'Total residues loaded: {L_total}')

        # We need to know which residues belong to which chain.
        # The preprocessing merges chains in order: B first, then A.
        # Let's get the split from the original PDB.
        scaffold_len = 0
        epitope_len = 0
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21]
                    resnum = int(line[22:26].strip())
                    if chain == 'B':
                        scaffold_len = max(scaffold_len, resnum)
                    elif chain == 'A':
                        epitope_len = max(epitope_len, resnum)

        # Actually, preprocess_protein_structure renumbers. Let's find the split differently.
        # The structure has residue_index. Let's check.
        if hasattr(structure, 'keys'):
            print(f'Structure keys: {list(structure.keys())[:15]}')
            if 'residue_index' in structure:
                print(f'residue_index: {structure["residue_index"]}')

        # Build batch
        transform = get_transform([
            {'type': 'mask_region', 'regions': regions},
            {'type': 'merge_protein'},
            {'type': 'patch_protein'},
        ])

        batch = recursive_to(PaddingCollate()([transform(structure)]), device)
        gen_mask = batch['generate_flag'][0].bool()

        # Count residues per chain using asym_id or entity_id
        # In the merged structure, the order is scaffold first, then epitope
        # Let's use chain_id from structure if available
        chain_id_field = batch.get('chain_id')
        asym_id = batch.get('asym_id')
        entity_id = batch.get('entity_id')

        if asym_id is not None:
            asym = asym_id[0].cpu().numpy()
            scaffold_mask_res = (asym == 0)
            epitope_mask_res = (asym == 1)
        elif chain_id_field is not None:
            cid = chain_id_field[0]
            # cid might be string tensor or int tensor
            if hasattr(cid, 'numpy'):
                cid = cid.cpu().numpy()
            scaffold_mask_res = (cid == 0) | (cid == ord('B'))
            epitope_mask_res = ~scaffold_mask_res
        else:
            # Fallback: assume first half is scaffold, second half is epitope
            # Actually, preprocess puts scaffold first
            # Let's use the gen_mask: CDR residues are on the scaffold
            gen_mask_np = gen_mask.cpu().numpy()
            scaffold_mask_res = np.ones(L_total, dtype=bool)
            epitope_mask_res = np.zeros(L_total, dtype=bool)
            # Find where gen_mask is True → these are CDR → scaffold
            # Find the contiguous block that doesn't contain CDR → epitope
            print('WARNING: cannot determine chain split from structure fields')
            # Use residue_index or seq indexing
            if 'residue_index' in batch:
                ri = batch['residue_index'][0].cpu().numpy()
                print(f'residue_index: {ri}')
            return

        scaffold_idx = np.where(scaffold_mask_res)[0]
        epitope_idx = np.where(epitope_mask_res)[0]
        L_scaffold = len(scaffold_idx)
        L_epitope = len(epitope_idx)

        print(f'Scaffold residues: {L_scaffold}')
        print(f'Epitope residues: {L_epitope}')
        print(f'CDR design residues (gen_mask): {gen_mask.sum().item()}')

        # ── Hook attention ──
        attention_data = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # module is GAEncoder (not GABlock)
                # We need to hook each GABlock individually
                pass
            return hook

        # Hook each GABlock
        encoder = model.receiver.encoder
        hooks = []
        captured_alphas = []

        for block_idx, block in enumerate(encoder.blocks):
            original_forward = block.forward

            def hooked_forward(self, R, t, x, z, mask, _bi=block_idx):
                # Call original but intercept intermediates
                # We need alpha - let's trace
                result = original_forward(R, t, x, z, mask)
                return result

            # Actually, let's use a PyTorch forward hook on the softmax output
            # Better: hook the _alpha_from_logits call
            # Simplest: monkey-patch the block to save alpha

            import types
            block._captured_alpha = None

            def capture_forward(self, R, t, x, z, mask, _orig=original_forward):
                # Replicate the forward but save alpha
                logits_node = self._node_logits(x)
                logits_pair = self._pair_logits(z)
                logits_spatial = self._spatial_logits(R, t, x)
                logits_sum = logits_node + logits_pair + logits_spatial

                from disorderflow.modules.encoders.ga import _alpha_from_logits
                alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)

                # Save alpha for analysis
                self._captured_alpha = alpha.detach().cpu()

                # Continue with original logic
                feat_p2n = self._pair_aggregation(alpha, z)
                feat_node = self._node_aggregation(alpha, x)
                feat_spatial = self._spatial_aggregation(alpha, R, t, x)
                feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))
                from disorderflow.modules.common.layers import mask_zero
                feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
                x_updated = self.layer_norm_1(x + feat_all)
                x_updated = self.dropout(x_updated)
                x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
                return x_updated

            block.forward = types.MethodType(capture_forward, block)

        # ── Run forward ──
        print('\nRunning forward pass with attention capture...')
        with torch.no_grad():
            # We need to run the receiver forward, not the full sample.
            # Build receiver inputs from the batch
            from disorderflow.modules.bfn.core import AntibodyBFN_Core
            # The core forward is complex. Let's just run the encoder directly.

            # Get the preprocessed features from the receiver
            # We need theta_seq, theta_pos, etc.
            # Actually, let's use the model's forward with a simple call

            # Simpler approach: just use the encoder directly
            # Get features from the batch
            # The batch has aa, pos_heavyatom, mask_heavyatom, etc.

            # Actually, let's go through the receiver forward
            receiver = model.receiver
            N = 1
            L = L_total

            # Create inputs
            theta_seq = torch.randn(N, L, 20, device=device)  # random sequence logits
            theta_pos = batch['pos_heavyatom'][:, :, 1, :].to(device)  # CA positions
            theta_ori = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(N, L, 3, 3).contiguous()
            theta_ori = theta_ori.reshape(N, L, 9)
            theta_ang = torch.zeros(N, L, 8, device=device)  # 4 angles * 2 (sin/cos)
            t_tensor = torch.ones(N, 1, device=device) * 0.5  # time step
            mask_res = batch['mask_heavyatom'][:, :, 0].to(device).float()

            # Build pair features from structure
            # pos_heavyatom: (N, L, 4, 3)
            ca_pos = batch['pos_heavyatom'][:, :, 1, :].to(device)
            # Simple pair feature: distance
            ca_dist = torch.cdist(ca_pos, ca_pos)  # (N, L, L)
            pair_feat = ca_dist.unsqueeze(-1).expand(N, L, L, 128).contiguous()

            backbone_pos = batch['pos_heavyatom'].to(device)

            print(f'theta_seq: {theta_seq.shape}, pos: {theta_pos.shape}, pair: {pair_feat.shape}')
            print(f'mask: {mask_res.shape}, mask sum: {mask_res.sum().item()}')

            # Run receiver forward
            _ = receiver(
                theta_seq, theta_pos, theta_ori, theta_ang,
                t_tensor, pair_feat, mask_res,
                backbone_pos=backbone_pos,
                prev_conf=None, prev_iptm=None, prev_pae=None,
                mask_gen=gen_mask.unsqueeze(0),
            )

        # ── Analyze attention ──
        print(f'\n{"=" * 70}')
        print(f'  ATTENTION ANALYSIS: Scaffold -> Epitope')
        print(f'{"=" * 70}')
        print(f'  {"Layer":<8s} {"Scaff->Self":<16s} {"Scaff->Epi":<16s} {"CDR->Epi":<16s} {"Epi->Epi":<16s}')
        print(f'  {"─"*8} {"─"*16} {"─"*16} {"─"*16} {"─"*16}')

        for block_idx, block in enumerate(encoder.blocks):
            alpha = block._captured_alpha
            if alpha is None:
                print(f'  Layer{block_idx:<3d}  NO DATA')
                continue

            # alpha: (N, L, L, n_heads) → average over heads
            alpha_avg = alpha.mean(dim=-1)[0].numpy()  # (L, L)

            # Scaffold (source) → Epitope (target)
            scaff_to_self = alpha_avg[scaffold_idx][:, scaffold_idx].mean()
            scaff_to_epi = alpha_avg[scaffold_idx][:, epitope_idx].mean()

            # CDR specifically → Epitope
            cdr_indices_0based = [i - 1 for i in cdr_indices]  # convert to 0-based
            # Filter to valid scaffold range
            cdr_in_scaffold = [i for i in cdr_indices_0based if i < L_scaffold]
            if cdr_in_scaffold:
                cdr_to_epi = alpha_avg[cdr_in_scaffold][:, epitope_idx].mean()
            else:
                cdr_to_epi = 0.0

            # Epitope → Epitope (self-attention within epitope)
            epi_to_epi = alpha_avg[epitope_idx][:, epitope_idx].mean()

            ratio = scaff_to_epi / max(scaff_to_self, 1e-9)
            cdr_ratio = cdr_to_epi / max(scaff_to_self, 1e-9)

            print(f'  Layer{block_idx:<3d}  {scaff_to_self:>14.6f}  {scaff_to_epi:>14.6f}  '
                  f'{cdr_to_epi:>14.6f}  {epi_to_epi:>14.6f}  '
                  f'(epi/self={ratio:.3f} cdr_epi/self={cdr_ratio:.3f})')

        # ── Summary ──
        final_alpha = encoder.blocks[-1]._captured_alpha
        if final_alpha is not None:
            final_avg = final_alpha.mean(dim=-1)[0].numpy()
            scaff_to_epi_final = final_avg[scaffold_idx][:, epitope_idx].mean()
            scaff_to_self_final = final_avg[scaffold_idx][:, scaffold_idx].mean()
            epi_fraction = scaff_to_epi_final / max(scaff_to_self_final + scaff_to_epi_final, 1e-9)

            print(f'\n  ── FINAL LAYER ──')
            print(f'  Scaffold -> Self:    {scaff_to_self_final:.6f}')
            print(f'  Scaffold -> Epitope: {scaff_to_epi_final:.6f}')
            print(f'  Epitope fraction:    {epi_fraction*100:.2f}%')
            print(f'  Scaffold self fraction: {(1-epi_fraction)*100:.2f}%')

            if epi_fraction < 0.01:
                print(f'\n  ❌ VERDICT: Epitope is INVISIBLE to scaffold.')
                print(f'     Scaffold attention to epitope < 1%')
                print(f'     BFN designs CDRs based on scaffold geometry alone.')
            elif epi_fraction < 0.05:
                print(f'\n  ⚠ VERDICT: Epitope is nearly invisible.')
                print(f'     Scaffold attention to epitope {epi_fraction*100:.1f}%')
            else:
                print(f'\n  ✓ VERDICT: Scaffold DOES attend to epitope.')
                print(f'     Epitope receives {epi_fraction*100:.1f}% of scaffold attention')

    finally:
        app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
        with open(cfg_path, 'w') as f:
            yaml.dump(app_cfg, f, default_flow_style=False)

    return


if __name__ == '__main__':
    run_attention_probe()
