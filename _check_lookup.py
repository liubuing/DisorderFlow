import pickle, numpy as np
lookup = pickle.load(open('data/sabdab_disorder_lookup.pkl', 'rb'))
print(f"Entries: {len(lookup)}")
high = [(k, v) for k, v in lookup.items() if v.max() > 0.4]
print(f"Max disorder > 0.4: {len(high)}")
print(f"Max disorder > 0.3: {sum(1 for k,v in lookup.items() if v.max() > 0.3)}")
print(f"Max disorder > 0.5: {sum(1 for k,v in lookup.items() if v.max() > 0.5)}")
print(f"Max disorder > 0.7: {sum(1 for k,v in lookup.items() if v.max() > 0.7)}")

if high:
    # Show distribution for one high-disorder entry
    k, v = high[0]
    print(f"\nExample: {k}")
    print(f"  Length: {len(v)}")
    print(f"  Mean: {v.mean():.4f}")
    print(f"  Max: {v.max():.4f}")
    print(f"  Values > 0.3: {int(np.sum(v > 0.3))}/{len(v)}")
    print(f"  Values > 0.5: {int(np.sum(v > 0.5))}/{len(v)}")
    print(f"  First 10: {v[:10].round(4).tolist()}")

# Check all means
means = [v.mean() for v in lookup.values()]
print(f"\nAll entry means: min={min(means):.4f}, max={max(means):.4f}, median={np.median(means):.4f}")
print(f"Mean > 0.3: {sum(1 for m in means if m > 0.3)}")
print(f"Mean > 0.2: {sum(1 for m in means if m > 0.2)}")
