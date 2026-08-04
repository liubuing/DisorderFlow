import lmdb
print("lmdb OK")

env = lmdb.open("/mnt/c/biological/DisorderFlow/data/peptide_h3_temporal_split_v3/sealed_final.lmdb", subdir=False, readonly=True, lock=False, readahead=False)
with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    for key, _ in cursor:
        if key != b'__len__':
            count += 1
            if count <= 3:
                print(f"  key: {key.decode()}")
    print(f"  total records: {count}")
env.close()
print("sealed_final.lmdb OK")
