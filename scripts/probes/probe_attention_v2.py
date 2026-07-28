import os
#!/usr/bin/env python
"""Probe BFN encoder attention v2 — simplified, correct pipeline."""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import yaml, torch, types, numpy as np

CDR_SPEC = 'B:26-33,51-58,97-113'
V11_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'
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
    from disorderflow.datasets.protein import preprocess_protein_structure
    from disorderflow.utils.transforms import get_transform
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    from antibody_epitope_complex import position_epitope
    from disorderflow.utils.misc import seed_all
    import bfn_loader
    from bfn_loader import load_bfn

    seed_all(42)
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    model, config = load_bfn(device)
    model.eval()

    # ── Build Complex structure ──
    complex_info = position_epitope(
        'data/misfolding_targets/5IMK.pdb',
        'data/misfolding_targets/2NAO_model1_A_1-42.pdb',
        scaffold_chain='B', epitope_chain='A', distance=6.0,
    )

    # ── Parse CDR regions ──
    import re
    regions = {}
    for cid, spec in re.findall(r'([A-Za-z0-9]+):([0-9,\-\s]+)', CDR_SPEC):
        indices = []
        for seg in spec.split(','):
            seg = seg.strip()
            if '-' in seg:
                a, b = seg.split('-')
                indices.extend(range(int(a.strip()), int(b.strip()) + 1))
            else:
                indices.append(int(seg))
        regions[cid] = sorted(set(indices))

    # ── Preprocess ──
    structure = preprocess_protein_structure(complex_info['pdb_path'], chain_ids=['B', 'A'])
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), device)

    # ── Map chain assignments ──
    chain_id_list = batch['chain_id']  # list of tuples like ('B',)
    chain_map = {}
    for i, cid_tuple in enumerate(chain_id_list):
        c = cid_tuple[0] if isinstance(cid_tuple, tuple) else cid_tuple
        if c not in chain_map:
            chain_map[c] = []
        chain_map[c].append(i)

    scaffold_idx = chain_map.get('B', [])
    epitope_idx = chain_map.get('A', [])

    # Handle chain ' ' (water etc) — add to epitope for safety
    for c in chain_map:
        if c not in ('A', 'B'):
            epitope_idx.extend(chain_map[c])

    L_total = len(chain_id_list)
    L_scaffold = len(scaffold_idx)
    L_epitope = len(epitope_idx)

    print(f'Total residues: {L_total}')
    print(f'Scaffold (B): {L_scaffold} residues [0-{max(scaffold_idx) if scaffold_idx else 0}]')
    print(f'Epitope (A): {L_epitope} residues [{min(epitope_idx) if epitope_idx else 0}-{max(epitope_idx) if epitope_idx else 0}]')

    # Identify CDR residues within scaffold
    gen_mask = batch['generate_flag'][0].bool()
    cdr_idx = torch.where(gen_mask)[0].cpu().tolist()
    cdr_in_scaffold = [i for i in cdr_idx if i in scaffold_idx]
    print(f'CDR residues: {len(cdr_idx)} (in scaffold: {len(cdr_in_scaffold)})')

    # ── Hook encoder blocks ──
    encoder = model.bfn.receiver.encoder

    for block_idx, block in enumerate(encoder.blocks):
        orig_forward = block.forward

        def make_capture(idx):
            def capture_forward(self, R, t, x, z, mask):
                logits_node = self._node_logits(x)
                logits_pair = self._pair_logits(z)
                logits_spatial = self._spatial_logits(R, t, x)
                logits_sum = logits_node + logits_pair + logits_spatial

                from disorderflow.modules.encoders.ga import _alpha_from_logits
                alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)
                self._captured_alpha = alpha.detach().cpu()

                feat_p2n = self._pair_aggregation(alpha, z)
                feat_node = self._node_aggregation(alpha, x)
                feat_spatial = self._spatial_aggregation(alpha, R, t, x)
                from disorderflow.modules.common.layers import mask_zero
                feat_all = mask_zero(mask.unsqueeze(-1), self.out_transform(
                    torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)))
                x_updated = self.layer_norm_1(x + feat_all)
                x_updated = self.dropout(x_updated)
                x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
                return x_updated
            return capture_forward

        block.forward = types.MethodType(make_capture(block_idx), block)

    # ── Run model.forward() with the batch ──
    # model.forward() handles pair_feat construction and calls model.bfn()
    print(f'\nRunning model forward pass...')

    with torch.no_grad():
        # model.forward() builds pair_feat and calls bfn.forward()
        loss_dict = model(batch)

    # ── Analyze ──
    print(f'\n{"=" * 70}')
    print(f'  ATTENTION: Scaffold -> Epitope per Layer')
    print(f'{"=" * 70}')
    print(f'  {"Layer":<8} {"Scaff->Self":<14} {"Scaff->Epi":<14} {"CDR->Epi":<14} '
          f'{"Scaff->Epi%":<12} {"CDR->Epi%":<12}')
    print(f'  {"─"*8} {"─"*14} {"─"*14} {"─"*14} {"─"*12} {"─"*12}')

    for block_idx, block in enumerate(encoder.blocks):
        alpha = getattr(block, '_captured_alpha', None)
        if alpha is None:
            print(f'  L{block_idx:<7d} NO DATA')
            continue

        alpha_avg = alpha.mean(dim=-1)[0].numpy()  # (L, L)

        s2s = alpha_avg[np.ix_(scaffold_idx, scaffold_idx)].mean()
        s2e = alpha_avg[np.ix_(scaffold_idx, epitope_idx)].mean()

        if cdr_in_scaffold:
            c2e = alpha_avg[np.ix_(cdr_in_scaffold, epitope_idx)].mean()
        else:
            c2e = 0.0

        total_scaff_attn = s2s + s2e
        s2e_pct = 100 * s2e / max(total_scaff_attn, 1e-9)
        c2e_pct = 100 * c2e / max(s2s + c2e, 1e-9)

        print(f'  L{block_idx:<7d} {s2s:>12.6f}  {s2e:>12.6f}  {c2e:>12.6f}  '
              f'{s2e_pct:>10.2f}%  {c2e_pct:>10.2f}%')

    # ── Final layer verdict ──
    final_alpha = getattr(encoder.blocks[-1], '_captured_alpha', None)
    if final_alpha is not None:
        final_avg = final_alpha.mean(dim=-1)[0].numpy()
        s2e_final = final_avg[np.ix_(scaffold_idx, epitope_idx)].mean()
        s2s_final = final_avg[np.ix_(scaffold_idx, scaffold_idx)].mean()
        epi_frac = 100 * s2e_final / max(s2s_final + s2e_final, 1e-9)

        print(f'\n  ── FINAL VERDICT ──')
        print(f'  Epitope receives {epi_frac:.1f}% of scaffold attention in final layer')

        # Also check: does epitope self-attention exist?
        e2e = final_avg[np.ix_(epitope_idx, epitope_idx)].mean()
        print(f'  Epitope self-attention: {e2e:.6f}')

        if epi_frac < 1.0:
            print(f'\n  ❌ Epitope is INVISIBLE to scaffold (< 1% attention)')
            print(f'     BFN designs CDRs based on scaffold geometry alone.')
            print(f'     The epitope reads as background noise.')
        elif epi_frac < 3.0:
            print(f'\n  ⚠ Epitope is barely visible ({epi_frac:.1f}% attention)')
            print(f'     Scaffold-dominated design with marginal epitope influence.')
        elif epi_frac < 8.0:
            print(f'\n  ~ Epitope has weak influence ({epi_frac:.1f}% attention)')
            print(f'     Some epitope signal reaches scaffold, but scaffold dominates.')
        else:
            print(f'\n  ✓ Epitope is visible ({epi_frac:.1f}% attention)')
            print(f'     BFN DOES attend to epitope residues.')

finally:
    app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)
