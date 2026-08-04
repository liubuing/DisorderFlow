import json, sys
from pathlib import Path
data = json.loads(Path("/mnt/c/biological/DisorderFlow/results/publication/h3_t2.1_temporal_final_v1/results.json").read_text())
agg = data["aggregate"]
print(json.dumps(agg, indent=2))
print()
valid = [r for r in data["results"] if r.get("valid_t2_1")]
failed = [r for r in data["results"] if r.get("torsion_hit_tier") and not r.get("valid_t2_1")]
no_tier = [r for r in data["results"] if not r.get("torsion_hit_tier")]
print(f"Valid: {len(valid)}/{len(data['results'])}")
print(f"Failed validation: {len(failed)}")
print(f"No torsion tier: {len(no_tier)}")
if no_tier:
    for r in no_tier:
        print(f"  {r['id']}: no RMSD tier hit")
if valid:
    contacts = [r["metrics"]["mean_held_out_contact_recovery"] for r in valid]
    print(f"Contact recovery: mean={sum(contacts)/len(contacts):.4f}, min={min(contacts):.4f}, max={max(contacts):.4f}")
