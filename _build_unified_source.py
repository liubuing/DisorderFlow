"""Build a unified source LMDB from merged IDP queue for the conformation build script."""
import lmdb, pickle, json, os

with open('D:/biological/DisorderFlow/data/conformation_queue_merged.json') as f:
    queue = json.load(f)

entries = queue['entries']
print(f'Loading {len(entries)} entries from multiple sources...')

# Open all source LMDBs
sources = {}
for tag in ['v10', 'idp_v3']:
    if tag == 'v10':
        path = 'D:/biological/DisorderFlow/data/confidence_merged_v10/confidence_train.lmdb'
    else:
        path = 'D:/biological/DisorderFlow/data/confidence_idp_v3'
    sources[tag] = lmdb.open(path, readonly=True, max_readers=1)

# Create unified destination
dst_path = 'D:/biological/DisorderFlow/data/confidence_idp_unified'
os.makedirs(dst_path, exist_ok=True)
# Remove old if exists
import shutil
if os.path.exists(os.path.join(dst_path, 'data.mdb')):
    shutil.rmtree(dst_path)
    os.makedirs(dst_path)

map_size = 10 * 1024 * 1024 * 1024  # 10 GB pre-allocated
dst_env = lmdb.open(dst_path, map_size=map_size)

written = 0
with dst_env.begin(write=True) as txn:
    for i, entry in enumerate(entries):
        src_env = sources[entry['source']]
        src_idx = entry['index']
        with src_env.begin() as src_txn:
            val = src_txn.get(f'{src_idx:08d}'.encode())
        if val is None:
            print(f'  MISSING: {entry["pdb_id"]} idx={src_idx} src={entry["source"]}')
            continue
        txn.put(f'{i:08d}'.encode(), val)
        written += 1

dst_env.close()
for env in sources.values():
    env.close()

print(f'Unified source LMDB: {written} entries at {dst_path}')
print(f'Ready for conformation build.')
