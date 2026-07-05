import os
#!/usr/bin/env python
"""Check if epitope residue identities actually change BFN internal representations.

Runs model.forward() on two complexes with same backbone but different epitope AA,
then compares internal feature differences.
"""
import sys, os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))

import yaml, torch, tempfile, shutil, numpy as np
from disorderflow.datasets.protein import preprocess_protein_structure
from disorderflow.utils.transforms import get_transform
from disorderflow.utils.data import PaddingCollate
from disorderflow.utils.train import recursive_to
from antibody_epitope_complex import position_epitope
from disorderflow.utils.misc import seed_all

V11_CKPT = 'logs/bfn_v11_seqconf_xpu_2026_06_19__23_01_18/checkpoints/best.pt'
CDR_SPEC = 'B:26-33,51-58,97-113'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def create_variant_pdb(template_pdb, chain_id, aa_func, output_path):
    with open(template_pdb) as f:
        lines = f.readlines()
    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith('ATOM') and line[21] == chain_id:
                old_res = line[17:20].strip()
                new_res = aa_func(old_res)
                if new_res != old_res:
                    line = line[:17] + f'{new_res:>3s}' + line[20:]
            f.write(line)

def build_and_run(epitope_pdb):
    """Build complex and run model.forward(), returning encoder features."""
    complex_info = position_epitope(
        'data/misfolding_targets/5IMK.pdb', epitope_pdb,
        scaffold_chain='B', epitope_chain='A', distance=6.0,
    )
    structure = preprocess_protein_structure(complex_info['pdb_path'], chain_ids=['B', 'A'])
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    batch = recursive_to(PaddingCollate()([transform(structure)]), device)

    # Run forward
    with torch.no_grad():
        _ = model(batch)

    # Extract encoder output features from each layer
    features = {}
    for block_idx, block in enumerate(model.bfn.receiver.encoder.blocks):
        alpha = getattr(block, '_captured_alpha', None)
        if alpha is not None:
            features[f'alpha_L{block_idx}'] = alpha.mean(dim=-1)[0].cpu()  # (L, L)

    # Cleanup
    tmp_dir = os.path.dirname(complex_info['pdb_path'])
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return features


# ── Setup ──
cfg_path = os.path.join(PROJECT_ROOT, 'app_config.yaml')
with open(cfg_path) as f:
    app_cfg = yaml.safe_load(f)
orig_ckpt = app_cfg['models']['bfn']['checkpoint']
app_cfg['models']['bfn']['checkpoint'] = V11_CKPT
with open(cfg_path, 'w') as f:
    yaml.dump(app_cfg, f, default_flow_style=False)

try:
    import bfn_loader
    bfn_loader._bfn_model = None
    bfn_loader._bfn_config = None
    seed_all(42)

    from bfn_loader import load_bfn
    model, config = load_bfn(device)
    model.eval()

    # Hook encoder
    from disorderflow.modules.encoders.ga import _alpha_from_logits
    import types

    for block_idx, block in enumerate(model.bfn.receiver.encoder.blocks):
        orig_forward = block.forward
        def make_capture(orig):
            def capture_forward(self, R, t, x, z, mask):
                logits_node = self._node_logits(x)
                logits_pair = self._pair_logits(z)
                logits_spatial = self._spatial_logits(R, t, x)
                logits_sum = logits_node + logits_pair + logits_spatial
                import numpy as np2
                alpha = _alpha_from_logits(logits_sum * np2.sqrt(1/3), mask)
                self._captured_alpha = alpha.detach().cpu()
                # Continue original
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
        block.forward = types.MethodType(make_capture(orig_forward), block)

    # ── Test: Native vs All-Ala epitope ──
    native_pdb = 'data/misfolding_targets/2NAO_model1_A_1-42.pdb'
    tmpdir = tempfile.mkdtemp()
    ala_pdb = os.path.join(tmpdir, 'all_ala.pdb')
    create_variant_pdb(native_pdb, 'A', lambda r: 'ALA', ala_pdb)

    print('Running native epitope...')
    feat_native = build_and_run(native_pdb)
    print('Running All-Ala epitope...')
    feat_ala = build_and_run(ala_pdb)

    # ── Compare ──
    print(f'\n{"=" * 70}')
    print(f'  EPITOPE SIGNAL TEST: Native vs All-Ala attention comparison')
    print(f'{"=" * 70}')

    for key in sorted(feat_native.keys()):
        native_alpha = feat_native[key]
        ala_alpha = feat_ala[key]
        diff = (native_alpha - ala_alpha).abs()

        # Per-region analysis
        L = native_alpha.shape[0]
        # scaffold: indices 0-103, epitope: 104-223
        scaff_idx = list(range(0, 104))
        epi_idx = list(range(104, L))

        # Attention from scaffold to epitope
        native_s2e = native_alpha[np.ix_(scaff_idx, epi_idx)].mean()
        ala_s2e = ala_alpha[np.ix_(scaff_idx, epi_idx)].mean()
        diff_s2e = diff[np.ix_(scaff_idx, epi_idx)].mean()

        # Overall attention difference
        total_diff = diff.mean()
        max_diff = diff.max()

        print(f'  {key}: native s->e={native_s2e:.6f}  ala s->e={ala_s2e:.6f}  '
              f'diff s->e={diff_s2e:.6f}  total_diff={total_diff:.6f}  max_diff={max_diff:.6f}')

    # Final answer
    final_key = sorted(feat_native.keys())[-1]
    final_diff = (feat_native[final_key] - feat_ala[final_key]).abs()
    max_final_diff = final_diff.max().item()
    mean_final_diff = final_diff.mean().item()

    print(f'\n  ── VERDICT ──')
    print(f'  Final layer attention difference: mean={mean_final_diff:.6f}  max={max_final_diff:.6f}')
    if max_final_diff < 1e-4:
        print(f'  ❌ Zero difference. Epitope AA identity has NO effect on attention.')
        print(f'     BFN\'s attention mechanism ignores epitope residue chemistry entirely.')
    elif max_final_diff < 0.001:
        print(f'  ⚠ Near-zero difference. Epitope AA identity has negligible effect.')
    else:
        print(f'  ✓ Epitope AA identity DOES change attention patterns.')
        print(f'    Max attention weight difference: {max_final_diff:.4f}')

finally:
    app_cfg['models']['bfn']['checkpoint'] = orig_ckpt
    with open(cfg_path, 'w') as f:
        yaml.dump(app_cfg, f, default_flow_style=False)
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
