"""Debug Complex mode: trace context chain handling in build_region_batch."""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'modules')
os.environ['DISORDERFLOW_CHECKPOINT'] = 'logs/bfn_v20_amplify_xpu_2026_07_02__21_32_16_v20_win/checkpoints/best.pt'

from Bio import PDB
from bfn_loader import build_region_batch, parse_region_spec

# Find the complex PDB
complex_dir = 'oc_validation_results/_complexes'
for f in os.listdir(complex_dir):
    if '5IMK_2NAO' in f:
        pdb_path = os.path.join(complex_dir, f)
        break
else:
    print("No complex PDB found!")
    sys.exit(1)

print(f"Complex PDB: {pdb_path}")

# Check what chains are in the PDB
parsed = PDB.PDBParser(QUIET=True).get_structure('x', pdb_path)
chains = [c.id for c in parsed[0].get_chains()]
print(f"Chains in PDB: {chains}")

# Check what residues are in each chain
for chain_id in chains:
    residues = list(parsed[0][chain_id].get_residues())
    if residues:
        first = residues[0].get_id()
        last = residues[-1].get_id()
        print(f"  Chain {chain_id}: {len(residues)} residues, first={first}, last={last}")

# CDR spec
region_spec = "B:26-33,51-58,97-113"
print(f"\nCDR spec: {region_spec}")

# Parse CDR spec
regions = parse_region_spec(region_spec)
design_chains = list(regions)
print(f"Design chains: {design_chains}")
print(f"Regions: {regions}")

# Test build_region_batch with context_chains=None (Complex mode)
print("\n=== Complex mode (context_chains=None) ===")
try:
    from disorderflow.utils.transforms import get_transform
    from disorderflow.datasets.protein import preprocess_protein_structure
    
    all_chains = sorted(
        set(design_chains + [c for c in chains if c not in design_chains]),
        key=lambda c: design_chains.index(c) if c in design_chains else 999)
    
    print(f"All chains (design first): {all_chains}")
    
    structure = preprocess_protein_structure(pdb_path, chain_ids=all_chains)
    print(f"Structure chains: {list(structure.keys()) if structure else 'None'}")
    
    transform = get_transform([
        {'type': 'mask_region', 'regions': regions},
        {'type': 'merge_protein'},
        {'type': 'patch_protein'},
    ])
    processed = transform(structure)
    print(f"Processed keys: {list(processed.keys())[:10]}")
    
    # Check chain_nb (which chain is which)
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.train import recursive_to
    import torch
    batch = recursive_to(PaddingCollate()([processed]), 'cpu')
    cn = batch['chain_nb'][0]
    unique_chains = sorted(set(int(x) for x in cn.tolist()))
    print(f"Unique chain numbers: {unique_chains}")
    for cnb in unique_chains:
        mask = (cn == cnb)
        n_res = int(mask.sum())
        print(f"  chain_nb {cnb}: {n_res} residues")
    
    # Check generate_flag
    gf = batch.get('generate_flag')
    if gf is not None:
        print(f"Generate flag sum: {int(gf.sum())}")
        for cnb in unique_chains:
            mask = (cn == cnb)
            n_gen = int((gf * mask).sum())
            print(f"  chain_nb {cnb}: {n_gen} generated")
    
    print("Complex mode build_region_batch OK")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
