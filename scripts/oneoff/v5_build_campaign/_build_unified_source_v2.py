"""
Build unified source LMDB from ALL v10 entries (no IDP pre-filtering).
AF2 multi-seed RMSF is the disorder measurement — we don't need pre-labeled IDP data.
Filter: L <= 500 only.
"""
# v5 构建活动的一次性脚本：把 v10 合并数据集的全部条目（不做 IDP 预筛选，
# 无序测量由 AF2 多种子 RMSF 提供）按长度 ≤500 过滤后，按长度升序复制为
# 统一源 LMDB（map_size 预留 20GB），并生成按新下标编号的处理队列 JSON，
# 供断点续跑的构建进程消费。
import lmdb, pickle, json, os, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / 'data/confidence_merged_v10/confidence_train.lmdb'
MAX_LEN = 500  # 长度上限：超过 500 残基的条目不参与 v5 构建

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
# 统一目标 LMDB：已存在则整目录删除重建；值按源 LMDB 原样字节拷贝
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
# 处理队列：新下标 → 源长度/PDB ID，构建进程按此队列断点续跑
queue = {
    'entries': [{'index': i, 'length': e['length'], 'pdb_id': e['pdb_id']} for i, e in enumerate(entries)],
    'total': written,
    'max_len': MAX_LEN,
}
with open(PROJECT_ROOT / 'data/conformation_queue_full.json', 'w') as f:
    json.dump(queue, f, indent=2)
print(f'Queue saved. Ready for build.')
