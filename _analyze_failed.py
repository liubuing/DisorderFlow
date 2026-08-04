import json
from pathlib import Path

base = Path("/mnt/c/biological/DisorderFlow/results/publication/h3_t2.1_temporal_final_v1/work")

print("=== Failed Cluster Analysis ===\n")

for work_dir in sorted(base.iterdir()):
    if not work_dir.is_dir():
        continue
    rr = work_dir / "record_result.json"
    if not rr.exists():
        continue
    result = json.loads(rr.read_text())
    rid = result["id"]
    valid = result.get("valid_t2_1", False)
    if valid:
        continue
    
    audit = result.get("t2_1_audit", {})
    arms = audit.get("arms", {})
    sc = arms.get("supplied_contacts", {})
    
    if isinstance(sc, dict) and "replicas" in sc:
        replicas = sc["replicas"]
        n_acc = sum(1 for r in replicas if r.get("accepted"))
        n_tot = len(replicas)
        print(f"[{rid}] accepted={n_acc}/{n_tot} scale={result.get('torsion_scale_degrees')}")
        for r in replicas:
            m = r.get("final_metrics", {})
            err = r.get("error", "")
            if m:
                ab = m.get("antibody_backbone_rmsd", "?")
                pp = m.get("peptide_backbone_rmsd", "?")
                clash = m.get("nonlocal_clashes_lt_1_5A", "?")
                contact = m.get("held_out_contact_retention", "?")
                acc = r.get("accepted", "?")
                issues = []
                if isinstance(ab, (int, float)) and ab > 0.5:
                    issues.append(f"abRMSD={ab:.2f}")
                if isinstance(pp, (int, float)) and pp > 3.5:
                    issues.append(f"pepRMSD={pp:.2f}")
                if clash and clash > 0:
                    issues.append(f"clashes={clash}")
                flag = "OK" if acc else f"REJECT({','.join(issues) if issues else 'unknown'})"
                print(f"  seed {r['seed']}: {flag} | contact={contact}")
            elif err:
                print(f"  seed {r['seed']}: ERROR={err}")
    elif isinstance(sc, dict) and "error" in sc:
        print(f"[{rid}] ARM ERROR: {sc['error']}")
    else:
        print(f"[{rid}] no replica data")
    print()
