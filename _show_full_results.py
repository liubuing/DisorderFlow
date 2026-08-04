import json
from pathlib import Path
data = json.loads(Path("/mnt/c/biological/DisorderFlow/results/publication/h3_t2.1_temporal_final_v1/results.json").read_text())

print("=== T2.1 Full Results ===")
print(f"Clusters: {len(data['results'])}")
print(f"Torsion hits: {len([r for r in data['results'] if r['torsion_hit_tier']])}")
print(f"Valid: {sum(1 for r in data['results'] if r.get('valid_t2_1'))}")
print()

for r in data["results"]:
    v = r.get("valid_t2_1", False)
    t = r.get("torsion_hit_tier", False)
    scale = r.get("torsion_scale_degrees", "N/A")
    rmsd = r.get("initial_peptide_rmsd", "N/A")
    m = r.get("metrics", {})
    contact = m.get("mean_held_out_contact_recovery", "N/A")
    rmsd_rec = m.get("mean_rmsd_recovery_angstrom", "N/A")
    status = "PASS" if v else ("TIER" if t else "NO_TIER")
    print(f"{status:8s} {r['id']:<22s} unit={r.get('unit','?'):6s} scale={str(scale):>5s} initRMSD={str(rmsd):>6s} contact={str(contact):>6s} rmsdRec={str(rmsd_rec):>7s}")

print()
print(f"Gate: valid_cluster_fraction >= 0.8")
print(f"Gate: held_out_contact_recovery CI95 lower > 0")
print(f"Gate: positive_fraction >= 0.7")
print(f"Result: {data['aggregate']['development_pass']}")
