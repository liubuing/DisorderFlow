"""检查源 LMDB 条目里 torch tensor 的实际类型"""
import sys, pickle, types
sys.path.insert(0, "/mnt/c/biological/DisorderFlow/modules")

import lmdb
e = lmdb.open("data/confidence_unified_v2", readonly=True, lock=False)
t = e.begin()
import pickle
val = t.get(b"00000000")
entry = pickle.loads(val)
print("top keys:", list(entry.keys()))
print("pdb_id:", entry.get("pdb_id"))
print("seq len:", len(entry.get("sequence", "")))
batch = entry.get("batch")
print("batch type:", type(batch))
if isinstance(batch, dict):
    print("batch keys:", list(batch.keys())[:20])
    for k in list(batch.keys())[:5]:
        v = batch[k]
        print(f"  batch[{k!r}] type={type(v).__name__} ", end="")
        try:
            print(f"shape/len={getattr(v,'shape',len(v) if hasattr(v,'__len__') else '?')}")
        except Exception:
            print()
e.close()
