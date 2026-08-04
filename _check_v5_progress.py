import lmdb, pickle
e = lmdb.open("data/confidence_conformation_v5/confidence_train.lmdb", readonly=True, lock=False)
t = e.begin()
v = t.get(b"__len__")
print("__len__:", pickle.loads(v) if v else 0)
keys = [k for k, _ in t.cursor() if k != b"__len__"]
print("actual keys:", len(keys))
if keys:
    print("last 3:", [k.decode() for k in keys[-3:]])
    sample = pickle.loads(t.get(keys[-1]))
    print("last entry pdb_id:", sample.get("pdb_id"), "L:", len(sample.get("sequence", "")), "n_conf:", sample.get("n_conformations"))
e.close()
