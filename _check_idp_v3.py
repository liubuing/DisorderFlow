"""Check idp_v3 LMDB contents"""
import lmdb, pickle

SRC = 'C:/biological/DisorderFlow/data/confidence_idp_v3/data.mdb'
# Actually idp_v3 is at a different path
import os
for p in ['C:/biological/DisorderFlow/data/confidence_idp_v3']:
    if os.path.isdir(p):
        env = lmdb.open(p, readonly=True, max_readers=1)
        n = env.stat()['entries']
        print(f'{p}: {n} entries')

        # Check a few entries
        for i in [0, 1, n//2, n-1]:
            with env.begin() as txn:
                val = txn.get(f'{i:08d}'.encode())
            if val:
                data = pickle.loads(val)
                seq = data.get('sequence', '')
                dis_frac = data.get('disorder_fraction', 'N/A')
                dis_src = data.get('disorder_source', 'N/A')
                pdb = data.get('pdb_id', '?')
                print(f'  [{i}] {pdb} L={len(seq)} dis_frac={dis_frac} src={dis_src}')
        env.close()
