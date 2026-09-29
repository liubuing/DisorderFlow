"""
Build unified source LMDB from ALL v10 entries (no IDP pre-filtering).
AF2 multi-seed RMSF is the disorder measurement — we don't need pre-labeled IDP data.
Filter: L <= 500 only.
"""
import lmdb, pickle, json, os, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / 'data/confidence_merged_v10/confidence_train.lmdb'
MAX_LEN = 500

env = lmdb.open(SRC, readonly=True, max_readers=1)
total = env.stat()['entries']
print(f'Scanning {total} entries from v10...')

entries = []
skipped_long = 0
for i in range(total):
    with env.begin() as txn:
        val = txn.get(f'{i:08d}'.encode())
    if val is None:
        continue
    data = pickle.loads(val)
    seq = data.get('sequence', '')
    L = len(seq)
    if L > MAX_LEN:
        skipped_long += 1
        continue
    entries.append({'index': i, 'length': L, 'pdb_id': data.get('pdb_id', '?')})

env.close()

entries.sort(key=lambda x: x['length'])
print(f'Entries L <= 500: {len(entries)} (skipped {skipped_long} long)')
print(f'Length range: {entries[0]["length"]} - {entries[-1]["length"]}')

# Build unified destination
dst_path = PROJECT_ROOT / 'data/confidence_unified_v2'
if os.path.exists(os.path.join(dst_path, 'data.mdb')):
    shutil.rmtree(dst_path)
os.makedirs(dst_path, exist_ok=True)

map_size = 20 * 1024 * 1024 * 1024  # 20 GB
dst_env = lmdb.open(dst_path, map_size=map_size)

src_env = lmdb.open(SRC, readonly=True, max_readers=1)
written = 0
with dst_env.begin(write=True) as txn:
    for i, entry in enumerate(entries):
        with src_env.begin() as src_txn:
            val = src_txn.get(f'{entry["index"]:08d}'.encode())
        if val:
            txn.put(f'{i:08d}'.encode(), val)
            written += 1

src_env.close()
dst_env.close()

print(f'Unified source: {written} entries at {dst_path}')

# Save queue
queue = {
    'entries': [{'index': i, 'length': e['length'], 'pdb_id': e['pdb_id']} for i, e in enumerate(entries)],
    'total': written,
    'max_len': MAX_LEN,
}
with open(PROJECT_ROOT / 'data/conformation_queue_full.json', 'w') as f:
    json.dump(queue, f, indent=2)
print(f'Queue saved. Ready for build.')
