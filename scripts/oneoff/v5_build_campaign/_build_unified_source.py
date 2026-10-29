"""Build a unified source LMDB from merged IDP queue for the conformation build script."""
import lmdb, pickle, json, os

# 读取合并后的 IDP 构象队列：每个 entry 记录 (source, index)，指明该样本来自哪个源 LMDB 的哪条 key。
with open('D:/biological/DisorderFlow/data/conformation_queue_merged.json') as f:
    queue = json.load(f)

entries = queue['entries']
print(f'Loading {len(entries)} entries from multiple sources...')

# Open all source LMDBs
# 两个来源：v10 为通用置信度训练集，idp_v3 为 IDP 专项子集；全部只读打开。
# 源 LMDB 的 key 约定为 8 位零填充整数索引（如 '00000042'）。
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

# 预分配 10 GB 的 map_size，避免写入中途页空间不足导致 MapFullError。
map_size = 10 * 1024 * 1024 * 1024  # 10 GB pre-allocated
dst_env = lmdb.open(dst_path, map_size=map_size)

# 按队列顺序把源记录的字节值原样拷入目标 LMDB，并把 key 重编为 0..N-1 的新序号；
# 单条缺失只打印告警并跳过，不中断整体合并（保证一次构建产出完整可用的统一数据集）。
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
