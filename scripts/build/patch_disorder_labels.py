"""Patch v9 LMDB with real per-residue disorder labels from EBI Proteins API.

For IDP entries: fetch EBI Proteins API → parse MobiDB-lite disorder regions.
For folded entries: all-zeros mask.
Fallback for API failures: use af2_plddt < 50 as disorder indicator.

Output: confidence_merged_v10/ (train + val LMDBs with updated disorder_mask).
"""

import os, sys, pickle, json, time, urllib.request
import lmdb
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fetch_disorder_regions(uniprot_id, seq_len, retries=2):
    """Fetch per-residue disorder mask from EBI Proteins API.

    Returns (mask, source) where mask is a numpy array of shape (seq_len,)
    with 1.0 for disordered residues, 0.0 for ordered.
    source is 'ebi_mobidb_lite', 'af2_plddt_fallback', or 'failed'.
    """
    for attempt in range(retries):
        try:
            url = f'https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}'
            req = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as f:
                data = json.loads(f.read())

            features = data.get('features', [])
            mask = np.zeros(seq_len, dtype=np.float32)
            found = False

            for feat in features:
                if feat['type'] == 'REGION' and 'disorder' in feat.get('description', '').lower():
                    start = int(feat['begin']) - 1
                    end = min(int(feat['end']), seq_len)
                    mask[start:end] = 1.0
                    found = True

            if found:
                return mask, 'ebi_mobidb_lite'

            # No disorder regions found — try again
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f'    EBI API failed for {uniprot_id}: {e}')

    return None, 'failed'


def af2_plddt_fallback(af2_plddt, threshold=50.0):
    """Use AF2 pLDDT < threshold as disorder indicator."""
    plddt = np.array(af2_plddt, dtype=np.float32)
    mask = (plddt < threshold).astype(np.float32)
    return mask, 'af2_plddt_fallback'


def patch_lmdb(src_path, dst_path, split_name):
    """Read all entries from src LMDB, add real disorder labels, write to dst LMDB."""
    src_env = lmdb.open(src_path, readonly=True, lock=False)
    dst_env = lmdb.open(dst_path, map_size=src_env.info()['map_size'] * 2)

    with src_env.begin() as src_txn:
        n_entries = pickle.loads(src_txn.get(b'__len__'))
        entries = []
        for i in range(n_entries):
            entries.append(pickle.loads(src_txn.get(f'{i:08d}'.encode())))

    print(f'[{split_name}] {n_entries} entries loaded, {sum(1 for e in entries if e.get("is_idp", False))} IDP')

    n_idp = 0
    n_ebi = 0
    n_fallback = 0
    n_failed = 0

    for i, entry in enumerate(entries):
        seq_len = len(entry['sequence'])
        is_idp = entry.get('is_idp', False)

        if is_idp:
            n_idp += 1
            uniprot_id = entry.get('pdb_id', '')
            mask, source = fetch_disorder_regions(uniprot_id, seq_len)

            if mask is None:
                # Fallback: use AF2 pLDDT
                plddt = np.array(entry['af2_plddt'], dtype=np.float32)
                if len(plddt) > seq_len:
                    plddt = plddt[:seq_len]
                mask, source = af2_plddt_fallback(plddt)
                n_fallback += 1
            else:
                n_ebi += 1

            entry['disorder_mask'] = torch.from_numpy(mask)
            entry['disorder_fraction'] = float(mask.mean()) if mask.sum() > 0 else 0.0
            entry['disorder_source'] = source

            if i < 5 or (i % 20 == 0):
                frac = entry['disorder_fraction']
                print(f'  [{i}] {uniprot_id}: {frac*100:.1f}% disordered ({source})')
        else:
            # Folded entry: all ordered
            mask = np.zeros(seq_len, dtype=np.float32)
            entry['disorder_mask'] = torch.from_numpy(mask)
            entry['disorder_fraction'] = 0.0
            entry['disorder_source'] = 'folded'

        # Delay between EBI API calls to avoid rate limiting
        if is_idp and i < n_entries - 1:
            time.sleep(0.3)

    # Write updated entries
    with dst_env.begin(write=True) as dst_txn:
        dst_txn.put(b'__len__', pickle.dumps(n_entries))
        for i, entry in enumerate(entries):
            dst_txn.put(f'{i:08d}'.encode(), pickle.dumps(entry))

    src_env.close()
    dst_env.close()

    print(f'[{split_name}] Done: {n_idp} IDP entries ({n_ebi} EBI, {n_fallback} fallback, {n_failed} failed)')
    return n_idp, n_ebi, n_fallback, n_failed


def main():
    base_src = 'data/confidence_merged_v9'
    base_dst = 'data/confidence_merged_v10'

    os.makedirs(f'{base_dst}/confidence_train.lmdb', exist_ok=True)
    os.makedirs(f'{base_dst}/confidence_val.lmdb', exist_ok=True)

    total_idp = total_ebi = total_fallback = 0

    for split in ['train', 'val']:
        src = f'{base_src}/confidence_{split}.lmdb'
        dst = f'{base_dst}/confidence_{split}.lmdb'
        n_idp, n_ebi, n_fallback, n_failed = patch_lmdb(src, dst, split)
        total_idp += n_idp
        total_ebi += n_ebi
        total_fallback += n_fallback

    # Write summary
    summary = {
        'source': 'v9 + EBI Proteins API per-residue disorder labels (MobiDB-lite)',
        'n_train': None,  # Will be read
        'n_val': None,
        'n_idp': total_idp,
        'n_idp_ebi': total_ebi,
        'n_idp_fallback': total_fallback,
    }
    with open(f'{base_dst}/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n=== Summary ===')
    print(f'  Total IDP entries: {total_idp}')
    print(f'  EBI MobiDB-lite:   {total_ebi}')
    print(f'  AF2 pLDDT fallback: {total_fallback}')
    print(f'  Output: {base_dst}/')


if __name__ == '__main__':
    main()
