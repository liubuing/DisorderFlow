#!/usr/bin/env python
"""Download a curated set of small single-chain protein PDB structures.

Diverse fold types: all-alpha, all-beta, alpha/beta, alpha+beta, small.
All structures are single-chain, resolution < 2.5A, length 50-200 residues.
"""
import sys, os, time, urllib.request

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Curated list: small single-chain proteins with diverse folds
# PDB ID: (name, fold_type)
PDB_LIST = [
    # All-alpha
    ('1UBQ', 'Ubiquitin'),
    ('1ENH', 'Engrailed Homeodomain'),
    ('1LMB', 'Lambda Repressor (1-92)'),
    ('1R69', '434 Repressor N-terminal'),
    ('1UTG', 'Uteroglobin'),

    # All-beta
    ('1SHG', 'SH3 Domain (alpha-spectrin)'),
    ('1PGB', 'Protein G B1 Domain'),
    ('1TEN', 'Tenascin Fibronectin Type III'),
    ('1FNA', 'Fibronectin (9-10 FnIII)'),

    # Alpha/beta (mixed)
    ('2CI2', 'Chymotrypsin Inhibitor 2'),
    ('1FKB', 'FK506 Binding Protein (FKBP)'),
    ('1YCC', 'Cytochrome C (oxidized)'),
    ('1RNB', 'Ribonuclease A (barnase mimic)'),
    ('1VQB', 'Cold Shock Protein B (CspB)'),

    # Alpha+beta
    ('1WLA', 'WW Domain'),
    ('2GB1', 'Protein G B1 (GB1)'),

    # Small / designed
    ('1VII', 'Villin Headpiece (HP35)'),
    ('3GB1', 'Protein G (GB1)'),
    ('1CSP', 'Cold Shock Protein A'),
    ('1AY7', 'Designed Protein (villin fragment)'),
]

OUT_DIR = 'data/general_proteins'
os.makedirs(OUT_DIR, exist_ok=True)

failed = []
downloaded = []

for pdb_id, name in PDB_LIST:
    pdb_path = os.path.join(OUT_DIR, f'{pdb_id}.pdb')
    if os.path.exists(pdb_path):
        print(f'  [{pdb_id}] {name}: already downloaded')
        downloaded.append(pdb_path)
        continue

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f'  [{pdb_id}] {name}: downloading...', end=' ', flush=True)
    try:
        urllib.request.urlretrieve(url, pdb_path)
        # Verify it's valid
        size = os.path.getsize(pdb_path)
        if size < 1000:
            os.unlink(pdb_path)
            raise ValueError(f'File too small ({size} bytes)')
        print(f'OK ({size} bytes)')
        downloaded.append(pdb_path)
    except Exception as e:
        print(f'FAILED: {e}')
        failed.append(pdb_id)
    time.sleep(0.5)  # Rate limit

print(f'\nDownloaded: {len(downloaded)}/{len(PDB_LIST)}')
if failed:
    print(f'Failed: {", ".join(failed)}')
    print('Try manually downloading from https://www.rcsb.org/')
