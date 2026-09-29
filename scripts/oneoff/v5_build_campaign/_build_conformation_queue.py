"""
Build a new conformation queue from v10 LMDB.
Filters: entries with disorder_fraction > 0 OR disorder_source='idp', length <= 500.
"""
import lmdb, pickle, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / 'data/confidence_merged_v10/confidence_train.lmdb'
MAX_LEN = 500

env = lmdb.open(SRC, readonly=True)
queue = []
total = env.stat()['entries']
print(f'Scanning {total} entries from v10...')

for i in range(total):
    key = f'{i:08d}'.encode()
    with env.begin() as txn:
        val = txn.get(key)
    if val is None:
        continue
    data = pickle.loads(val)

    seq = data.get('sequence', '')
    L = len(seq)
    if L > MAX_LEN:
        continue

    # Filter: keep IDP or partially disordered entries
    disorder_frac = data.get('disorder_fraction', 0)
    disorder_source = data.get('disorder_source', '')
    disorder_mask = data.get('disorder_mask', None)

    # Include if: has any disorder OR is explicitly IDP source
    is_disordered = (
        (isinstance(disorder_frac, (int, float)) and disorder_frac > 0) or
        disorder_source == 'idp' or
        (hasattr(disorder_mask, 'sum') and disorder_mask.sum() > 0)
    )

    if is_disordered:
        queue.append({'index': i, 'length': L, 'pdb_id': data.get('pdb_id', '?'),
                       'disorder_frac': float(disorder_frac)})

env.close()

# Sort by ascending length (shorter first = faster build startup)
queue.sort(key=lambda x: x['length'])

print(f'Queue: {len(queue)} entries (min L={queue[0]["length"]}, max L={queue[-1]["length"]})')
print(f'Sample first 5:')
for e in queue[:5]:
    print(f'  idx={e["index"]}, pdb={e["pdb_id"]}, L={e["length"]}, dis_frac={e["disorder_frac"]:.3f}')
print(f'Sample last 5:')
for e in queue[-5:]:
    print(f'  idx={e["index"]}, pdb={e["pdb_id"]}, L={e["length"]}, dis_frac={e["disorder_frac"]:.3f}')

# Save queue
out = {'indices': [e['index'] for e in queue],
       'lengths': [e['length'] for e in queue]}
with open(PROJECT_ROOT / 'data/conformation_queue_v10.json', 'w') as f:
    json.dump(out, f)

print(f'\nSaved to data/conformation_queue_v10.json')
print(f'Total entries to build: {len(queue)}')
