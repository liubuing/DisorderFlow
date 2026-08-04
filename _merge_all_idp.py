"""Merge all IDP entries from v8, v9, v10, idp_v3 into a single build queue."""
import lmdb, pickle, json

MAX_LEN = 500
seen = set()
all_entries = []

sources = [
    ('C:/biological/DisorderFlow/data/confidence_merged_v8/confidence_train.lmdb', 'v8'),
    ('C:/biological/DisorderFlow/data/confidence_merged_v9/confidence_train.lmdb', 'v9'),
    ('C:/biological/DisorderFlow/data/confidence_merged_v10/confidence_train.lmdb', 'v10'),
    ('C:/biological/DisorderFlow/data/confidence_idp_v3', 'idp_v3'),
]

for path, tag in sources:
    env = lmdb.open(path, readonly=True, max_readers=1)
    n = env.stat()['entries']
    count = 0
    for i in range(n):
        with env.begin() as txn:
            val = txn.get(f'{i:08d}'.encode())
        if val is None:
            continue
        data = pickle.loads(val)
        seq = data.get('sequence', '')
        L = len(seq)
        if L > MAX_LEN:
            continue

        pdb_id = data.get('pdb_id', '?')
        if pdb_id in seen:
            continue

        # IDP filter: idp_v3 entries are all IDP
        # v8/v9/v10: check disorder_fraction > 0 or disorder_source
        dis_frac = data.get('disorder_fraction', -1)
        dis_src = data.get('disorder_source', '')
        dis_mask = data.get('disorder_mask', None)

        is_idp = (
            tag == 'idp_v3' or
            (isinstance(dis_frac, (int, float)) and dis_frac > 0) or
            dis_src == 'idp' or
            (hasattr(dis_mask, 'sum') and dis_mask.sum() > 0)
        )

        if is_idp:
            seen.add(pdb_id)
            all_entries.append({'pdb_id': pdb_id, 'length': L, 'source': tag, 'index': i,
                                'disorder_frac': float(dis_frac) if isinstance(dis_frac, (int, float)) else 0.0})
            count += 1
    env.close()
    print(f'{tag}: {count} IDP entries (total={n})')

all_entries.sort(key=lambda x: x['length'])
print(f'\nTotal unique IDP entries (L <= {MAX_LEN}): {len(all_entries)}')
print(f'Length range: {all_entries[0]["length"]} - {all_entries[-1]["length"]}')

# Save merged queue
queue = {
    'entries': all_entries,
    'total': len(all_entries),
    'max_len': MAX_LEN,
}
with open('C:/biological/DisorderFlow/data/conformation_queue_merged.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f'Saved to data/conformation_queue_merged.json')
