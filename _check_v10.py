import lmdb, pickle

env = lmdb.open('C:/biological/DisorderFlow/data/confidence_merged_v10/confidence_train.lmdb', readonly=True)
with env.begin() as txn:
    val = txn.get(b'00000000')
    data = pickle.loads(val)
    print('Keys:', sorted(data.keys()))
    for k, v in sorted(data.items()):
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        elif isinstance(v, list):
            print(f'  {k}: list len={len(v)}')
        elif isinstance(v, str):
            print(f'  {k}: str len={len(v)}, preview={v[:80]}')
        else:
            print(f'  {k}: {type(v).__name__} = {str(v)[:120]}')

    # Check IDP status
    is_idp = data.get('is_idp', 'NOT FOUND')
    disorder = data.get('disorder_label', 'NOT FOUND')
    seq = data.get('sequence', 'NOT FOUND')
    print(f'\nis_idp: {is_idp}')
    print(f'disorder_label: {disorder.shape if hasattr(disorder, "shape") else disorder}')
    print(f'sequence: {seq[:80] if isinstance(seq, str) else "NOT FOUND"}')

env.close()
print(f'\nTotal entries: {env.stat()["entries"]}')
