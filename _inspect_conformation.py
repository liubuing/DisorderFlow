import lmdb, pickle, torch
import numpy as np

env = lmdb.open('C:/biological/DisorderFlow/data/confidence_conformation_v1/confidence_train.lmdb', readonly=True)
with env.begin() as txn:
    # Check a few entries
    for i in range(min(5, env.stat()['entries'])):
        key = f'{i:08d}'.encode()
        val = txn.get(key)
        data = pickle.loads(val)
        print(f'\n=== Entry {i}: {data.get("pdb_id", "?")} ===')
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k}: Tensor shape={v.shape}, dtype={v.dtype}, sample={v.flatten()[:5]}')
            elif isinstance(v, np.ndarray):
                print(f'  {k}: ndarray shape={v.shape}, dtype={v.dtype}, sample={v.flatten()[:5]}')
            elif isinstance(v, list):
                if v and isinstance(v[0], np.ndarray):
                    print(f'  {k}: list of {len(v)} ndarrays, shapes={[x.shape for x in v[:2]]}')
                else:
                    print(f'  {k}: list len={len(v)}, sample={str(v[:2])[:100]}')
            else:
                s = str(v)
                print(f'  {k}: {s[:150]}')

    # Count entries with rmsf vs disorder_label
    print('\n=== Checking which entries have rmsf/disorder_label ===')
    has_rmsf = 0
    has_disorder = 0
    for i in range(env.stat()['entries']):
        key = f'{i:08d}'.encode()
        val = txn.get(key)
        data = pickle.loads(val)
        if 'rmsf' in data: has_rmsf += 1
        if 'disorder_label' in data: has_disorder += 1
    print(f'Entries with rmsf: {has_rmsf}')
    print(f'Entries with disorder_label: {has_disorder}')
    print(f'Total entries: {env.stat()["entries"]}')

env.close()
