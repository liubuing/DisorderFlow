import pickle, numpy as np, os

# Check experimental lookup
lookup = pickle.load(open('data/sabdab_disorder_lookup_experimental.pkl', 'rb'))
if isinstance(lookup, dict):
    values = list(lookup.values())
    if values and isinstance(values[0], dict):
        # Might have metadata wrapper
        print(f"Keys: {list(lookup.keys())[:5]}")
        first = list(lookup.values())[0]
        print(f"First value type: {type(first)}")
        if isinstance(first, dict):
            print(f"First dict keys: {list(first.keys())[:5]}")
            # Assume 'profile' or array key
            for k in ('profile', 'disorder', 'values', 'labels'):
                if k in first:
                    print(f"Key '{k}' available, shape: {np.array(first[k]).shape}")
    else:
        profiles = [v for v in values if isinstance(v, np.ndarray)]
        if profiles:
            means = [p.mean() for p in profiles]
            maxes = [p.max() for p in profiles]
            print(f"Entries: {len(profiles)}")
            print(f"Mean range: [{min(means):.4f}, {max(means):.4f}], median: {np.median(means):.4f}")
            print(f"Max range: [{min(maxes):.4f}, {max(maxes):.4f}]")
            print(f"Mean > 0.3: {sum(1 for m in means if m > 0.3)}")
            print(f"Mean > 0.5: {sum(1 for m in means if m > 0.5)}")
            # Show one
            p = profiles[0]
            print(f"\nFirst entry: mean={p.mean():.4f}, max={p.max():.4f}, len={len(p)}")
            print(f"Values > 0.3: {int(np.sum(p > 0.3))}/{len(p)}")
            print(f"Values > 0.5: {int(np.sum(p > 0.5))}/{len(p)}")
