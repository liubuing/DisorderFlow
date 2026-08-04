"""Check DisProt experimental labels in confidence_idp_unified LMDB."""
import lmdb, pickle, sys
import numpy as np

env = lmdb.open('data/confidence_idp_unified', readonly=True, lock=False, readahead=False, subdir=True)
with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    with_labels = 0
    with_mask = 0
    mask_sums = []
    for key, value in cursor:
        if key == b'__len__':
            continue
        try:
            record = pickle.loads(value)
        except:
            continue
        count += 1
        if count > 200:
            break
        has_label = record.get('disorder_label') is not None
        has_mask = record.get('disorder_mask') is not None
        if has_label:
            with_labels += 1
        if has_mask:
            with_mask += 1
            d = record['disorder_mask']
            if hasattr(d, 'sum'):
                s = float(d.sum())
                l = len(d) if hasattr(d, '__len__') else 0
                mask_sums.append((s, l))

env.close()

print(f"Entries sampled: {count}")
print(f"With disorder_label: {with_labels}")
print(f"With disorder_mask: {with_mask}")
if mask_sums:
    sums = [s for s, l in mask_sums]
    lengths = [l for s, l in mask_sums]
    fractions = [s/l if l > 0 else 0 for s, l in mask_sums]
    print(f"Disorder mask fractions: min={min(fractions):.3f} max={max(fractions):.3f} median={np.median(fractions):.3f}")
    print(f"Entries with fraction > 0.2: {sum(1 for f in fractions if f > 0.2)}")
    print(f"Entries with fraction > 0.5: {sum(1 for f in fractions if f > 0.5)}")
    # Show first entry with significant disorder
    for i, (s, l) in enumerate(mask_sums):
        if l > 0 and s/l > 0.3:
            print(f"\nExample entry {i}: {l} residues, {s:.0f} disordered ({s/l:.1%})")
            break
