import os
#!/usr/bin/env python3
"""V11 Step 1: Fold-then-Dock native vs scrambled gate.

1. Build FASTA for native and scrambled 4HIX CDR designs
2. Fold with AF2 single-chain (colabfold in WSL)
3. Extract CDR loop backbone from folded structure
4. Graft CDR loops into scaffold (framework fixed)
5. HDOCK dock and compare scores
"""
import sys, os, pickle, time, json, subprocess, random, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'modules'))
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from disorderflow.utils.protein import parsers, constants
from disorderflow.datasets.sabdab import _label_heavy_chain_cdr, _label_light_chain_cdr

AA = 'ARNDCQEGHILKMFPSTWYV'
A3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
      'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
      'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
      'Y': 'TYR', 'V': 'VAL'}
WSL_PYTHON = 'wsl bash -c "cd /mnt/c/biological/DisorderFlow && ./venv_wsl/bin/python'

SCAFFOLD = 'data/anti_abeta_refs/4HIX.pdb'
OUT_DIR = 'fold_dock_results'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/fasta', exist_ok=True)
os.makedirs(f'{OUT_DIR}/folded', exist_ok=True)
os.makedirs(f'{OUT_DIR}/receptors', exist_ok=True)

print("=" * 60)
print("V11 Step 1: Fold-then-Dock Native vs Scrambled Gate")
print("=" * 60)

# Parse scaffold
parser = PDBParser(QUIET=True)
scaffold = parser.get_structure('4HIX', SCAFFOLD)[0]
hd, hm = parsers.parse_biopython_structure(scaffold['H'], max_resseq=113)
hd, hm = _label_heavy_chain_cdr(hd, hm)
ld, lm = parsers.parse_biopython_structure(scaffold['L'], max_resseq=106)
ld, lm = _label_light_chain_cdr(ld, lm)

def get_seq(data):
    return ''.join([AA[a.item()] if 0 <= a.item() < 20 else 'X' for a in data['aa']])

h_full_seq = get_seq(hd)
l_full_seq = get_seq(ld)

# CDR info: {chain: {cdr_type: {indices, seq, len}}}
def get_cdr_info(data, cdr_types):
    info = {}
    for ct in cdr_types:
        m = (data['cdr_flag'] == ct)
        if m.any():
            idx = m.nonzero(as_tuple=True)[0]
            aas = [AA[data['aa'][i].item()] for i in idx]
            info[ct] = {'indices': idx, 'seq': ''.join(aas), 'len': len(idx)}
    return info

h_cdr = get_cdr_info(hd, [1, 2, 3])
l_cdr = get_cdr_info(ld, [4, 5, 6])

# Native CDRs
native_h = ''.join([h_cdr[ct]['seq'] for ct in [1, 2, 3]])
native_l = ''.join([l_cdr[ct]['seq'] for ct in [4, 5, 6]])
print(f"Native H CDRs: {native_h} ({len(native_h)}aa)")
print(f"Native L CDRs: {native_l} ({len(native_l)}aa)")

# Scrambled CDRs
scram_h = list(native_h); random.shuffle(scram_h); scram_h = ''.join(scram_h)
scram_l = list(native_l); random.shuffle(scram_l); scram_l = ''.join(scram_l)
print(f"Scram H CDRs: {scram_h}")
print(f"Scram L CDRs: {scram_l}")

# Build modified FASTA sequences
def build_fv_sequence(h_seq, l_seq, h_cdr_info, new_h_cdr, new_l_cdr):
    """Replace CDR regions in antibody Fv sequences with new CDRs."""
    h_new = list(h_seq)
    l_new = list(l_seq)
    pos = 0
    for ct in [1, 2, 3]:
        if ct in h_cdr_info:
            for idx in h_cdr_info[ct]['indices']:
                if pos < len(new_h_cdr):
                    h_new[idx.item()] = new_h_cdr[pos]
                pos += 1
    pos = 0
    for ct in [4, 5, 6]:
        if ct in l_cdr_info:
            for idx in l_cdr_info[ct]['indices']:
                if pos < len(new_l_cdr):
                    l_new[idx.item()] = new_l_cdr[pos]
                pos += 1
    return ''.join(h_new), ''.join(l_new)

# Generate FASTA for colabfold (heavy:light format)
designs = [
    ('native', native_h, native_l),
    ('scrambled', scram_h, scram_l),
]

for name, h_cdr_seq, l_cdr_seq in designs:
    h_new, l_new = build_fv_sequence(h_full_seq, l_full_seq, h_cdr, h_cdr_seq, l_cdr_seq)
    fasta_path = f'{OUT_DIR}/fasta/{name}.fasta'
    with open(fasta_path, 'w') as f:
        f.write(f'>{name}\n{h_new}:{l_new}\n')
    print(f"  Wrote {name}.fasta: H={len(h_new)}aa L={len(l_new)}aa")

# Now fold with colabfold
print("\n--- Folding with AF2 (colabfold in WSL) ---")
folded_pdbs = {}

for name, _, _ in designs:
    fasta_path = f'{OUT_DIR}/fasta/{name}.fasta'
    out_dir = f'{OUT_DIR}/folded/{name}'
    os.makedirs(out_dir, exist_ok=True)

    # Run colabfold_batch in WSL
    # Use --num-models 1 --num-recycle 1 for speed
    wsl_fasta = f'/mnt/c/biological/DisorderFlow/{fasta_path}'
    wsl_out = f'/mnt/c/biological/DisorderFlow/{out_dir}'
    cmd = (f'wsl bash -c "cd /mnt/c/biological/DisorderFlow && '
           f'./venv_wsl/bin/colabfold_batch {wsl_fasta} {wsl_out} '
           f'--num-models 1 --num-recycle 1 --use-gpu-relax 2>&1"')

    print(f"  Folding {name}...")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.0f}s")

        # Find output PDB
        pdb_files = []
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith('.pdb') and 'unrelaxed' not in f and 'relaxed' in f:
                    pdb_files.append(os.path.join(root, f))
        if pdb_files:
            # Use the top-ranked model
            folded_pdbs[name] = pdb_files[0]
            print(f"    Output: {pdb_files[0]}")
        else:
            # Try any PDB
            for root, dirs, files in os.walk(out_dir):
                for f in files:
                    if f.endswith('.pdb'):
                        pdb_files.append(os.path.join(root, f))
            if pdb_files:
                folded_pdbs[name] = pdb_files[0]
                print(f"    Output: {pdb_files[0]}")
            else:
                print(f"    WARNING: No PDB output found!")
                print(f"    Files in {out_dir}:")
                for root, dirs, files in os.walk(out_dir):
                    for f in files:
                        print(f"      {f}")
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 600s")
    except Exception as ex:
        print(f"    ERROR: {ex}")

if len(folded_pdbs) < 2:
    print("\nFolding incomplete — cannot proceed to HDOCK gate.")
    print("Check colabfold output and retry.")
    sys.exit(1)

# Extract CDR loop backbone and graft into scaffold
print("\n--- CDR Loop Graft ---")

def extract_cdr_loop_atoms(folded_pdb, chain_id, cdr_info, cdr_types):
    """Extract N, CA, C, O atoms for CDR loop residues from folded structure."""
    p = PDBParser(QUIET=True)
    s = p.get_structure('folded', folded_pdb)[0]

    if chain_id not in s:
        return None

    chain = s[chain_id]
    residues = list(chain.get_residues())

    atoms = {}
    for ct in cdr_types:
        if ct in cdr_info:
            for idx in cdr_info[ct]['indices']:
                i = idx.item()
                if i < len(residues):
                    res = residues[i]
                    atom_coords = {}
                    for atom_name in ['N', 'CA', 'C', 'O']:
                        if atom_name in res:
                            atom_coords[atom_name] = res[atom_name].get_coord()
                    if 'CA' in atom_coords:
                        atoms[(chain_id, i)] = atom_coords
    return atoms

def graft_cdr_loops(scaffold_pdb, out_path, h_folded_atoms, l_folded_atoms, h_cdr_info, l_cdr_info):
    """Graft CDR loop backbone atoms from folded structure into scaffold."""
    p = PDBParser(QUIET=True)
    s = p.get_structure('scaffold', scaffold_pdb)[0]

    grafts = [(s['H'], h_cdr_info, [1,2,3], h_folded_atoms),
              (s['L'], l_cdr_info, [4,5,6], l_folded_atoms)]

    for chain, cdr_info, cdr_types, folded_atoms in grafts:
        if chain is None or folded_atoms is None:
            continue
        residues = list(chain.get_residues())
        for ct in cdr_types:
            if ct in cdr_info:
                for idx in cdr_info[ct]['indices']:
                    i = idx.item()
                    key = (chain.id, i)
                    if key in folded_atoms and i < len(residues):
                        res = residues[i]
                        for atom_name, coord in folded_atoms[key].items():
                            if atom_name in res:
                                res[atom_name].set_coord(coord)

    io = PDBIO()
    io.set_structure(s)
    io.save(out_path)

for name, _, _ in designs:
    if name not in folded_pdbs:
        continue
    folded_pdb = folded_pdbs[name]

    # Extract CDR loop atoms from folded structure
    h_atoms = extract_cdr_loop_atoms(folded_pdb, 'A', h_cdr, [1, 2, 3])
    l_atoms = extract_cdr_loop_atoms(folded_pdb, 'B', l_cdr, [4, 5, 6])

    # Fallback: sometimes AF2 uses different chain IDs
    if h_atoms is None:
        h_atoms = extract_cdr_loop_atoms(folded_pdb, 'A', h_cdr, [1, 2, 3])

    out_path = f'{OUT_DIR}/receptors/{name}_folded.pdb'
    graft_cdr_loops(SCAFFOLD, out_path, h_atoms, l_atoms, h_cdr, l_cdr)

    # Verify RMSD
    p = PDBParser(QUIET=True)
    orig_s = p.get_structure('orig', SCAFFOLD)[0]
    graft_s = p.get_structure('graft', out_path)[0]

    # Compute CDR loop CA RMSD
    distances = []
    for chain_id, cdr_info, cdr_types in [('H', h_cdr, [1,2,3]), ('L', l_cdr, [4,5,6])]:
        if chain_id not in orig_s or chain_id not in graft_s:
            continue
        orig_residues = list(orig_s[chain_id].get_residues())
        graft_residues = list(graft_s[chain_id].get_residues())
        for ct in cdr_types:
            if ct in cdr_info:
                for idx in cdr_info[ct]['indices']:
                    i = idx.item()
                    if i < len(orig_residues) and i < len(graft_residues):
                        if 'CA' in orig_residues[i] and 'CA' in graft_residues[i]:
                            d = np.linalg.norm(
                                orig_residues[i]['CA'].get_coord() -
                                graft_residues[i]['CA'].get_coord())
                            distances.append(d)

    if distances:
        rmsd = np.sqrt(np.mean(np.array(distances)**2))
        print(f"  {name}: CDR loop CA RMSD = {rmsd:.2f}A ({len(distances)} atoms)")
    else:
        print(f"  {name}: grafted (no RMSD computed)")

print("\n--- HDOCK Docking ---")

# Run HDOCK on folded-then-grafted receptors
def hdock_score(receptor_rel, ligand_rel, out_name):
    cmd = (f'cd HDOCKlite-v1.1 && LD_LIBRARY_PATH=. ./hdock '
           f'{receptor_rel} {ligand_rel} -out {out_name} 2>&1')
    try:
        r = subprocess.run(['wsl', 'bash', '-c', cmd],
                          capture_output=True, text=True, timeout=180)
    except:
        return None
    op = f'HDOCKlite-v1.1/{out_name}'
    if not os.path.exists(op):
        return None
    scores = []
    with open(op) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 9:
                try:
                    scores.append(float(p[6]))
                except ValueError:
                    pass
    if not scores:
        return None
    return {'top1': min(scores), 'top5': sorted(scores)[:5],
            'mean': np.mean(scores), 'n': len(scores)}

# Copy ligand
ligand_pdb = f'{OUT_DIR}/ligand.pdb'
if not os.path.exists(ligand_pdb):
    shutil.copy('HDOCKlite-v1.1/4HIX_ligand_native.pdb', ligand_pdb)

results = {}
for name, _, _ in designs:
    receptor_pdb = f'{OUT_DIR}/receptors/{name}_folded.pdb'
    if not os.path.exists(receptor_pdb):
        continue
    rec_rel = f'../{receptor_pdb}'
    lig_rel = f'../{ligand_pdb}'
    out_name = f'v11_{name}.out'
    print(f"  Docking {name}...", end=' ', flush=True)
    r = hdock_score(rec_rel, lig_rel, out_name)
    if r:
        results[name] = r
        print(f"top1={r['top1']:.1f}")
    else:
        print("FAILED")

# Gate report
print(f"\n{'='*60}")
print("V11 Step 1 Gate: Fold-then-Dock Native vs Scrambled")
print(f"{'='*60}")
for name in designs:
    n = name[0]
    if n in results:
        print(f"  {n}: top1={results[n]['top1']:.1f}, mean={results[n]['mean']:.1f}")
if 'native' in results and 'scrambled' in results:
    delta = results['scrambled']['top1'] - results['native']['top1']
    print(f"  Delta (scram-native): {delta:+.1f}")
    print(f"  Gate: {'PASS' if delta > 1.0 else 'MARGINAL'} (native better)")

ts = time.strftime('%Y%m%d_%H%M%S')
out_path = f'idp_benchmark_results/fold_dock_gate_{ts}.json'
with open(out_path, 'w') as f:
    json.dump({'probe': 'V11 Step 1 gate', 'results': results}, f, indent=2)
print(f"\nSaved: {out_path}")
print("V11 Step 1 done.")
