"""Build disorder lookup from AF2 pLDDT in IDP LMDB entries.
Uses 1-pLDDT as per-residue disorder score (validated proxy).
Output: data/sabdab_disorder_lookup_plddt.pkl
"""
import lmdb, pickle, numpy as np, torch

IDP_LMDB = 'data/confidence_idp_unified'
OUT = 'data/sabdab_disorder_lookup_plddt.pkl'

env = lmdb.open(IDP_LMDB, readonly=True, lock=False, readahead=False, subdir=True)
lookup = {}
seen = set()

with env.begin() as txn:
    for key, value in txn.cursor():
        if key == b'__len__':
            continue
        try:
            rec = pickle.loads(value)
        except:
            continue
        
        pdb_id = rec.get('pdb_id', key.decode())
        if pdb_id in seen:
            continue
        seen.add(pdb_id)
        
        plddt = rec.get('af2_plddt')
        if plddt is None:
            continue
        
        if isinstance(plddt, torch.Tensor):
            plddt = plddt.numpy()
        elif not isinstance(plddt, np.ndarray):
            continue
        
        # pLDDT is [0,100] → disorder = 1 - pLDDT/100
        disorder = 1.0 - np.clip(np.array(plddt, dtype=np.float32) / 100.0, 0, 1)
        lookup[pdb_id] = disorder

env.close()

with open(OUT, 'wb') as f:
    pickle.dump(lookup, f)

print(f"Built pLDDT-based disorder lookup: {len(lookup)} entries")
if lookup:
    k = list(lookup.keys())[0]
    v = lookup[k]
    print(f"  Example {k}: L={len(v)}, mean={v.mean():.3f}, "
          f"dis>0.3={int((v>=0.3).sum())}, dis>0.5={int((v>=0.5).sum())}")
