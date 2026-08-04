import sys, json, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import read_record, write_record_backbone
from generate_h3_peptide_t21 import generate_t2_1
from Bio.PDB import PDBParser

config = yaml.safe_load(Path(ROOT / "configs/benchmarks/peptide_h3_t2.1_temporal_final_v1.yml").read_text())
audit = json.loads(Path(ROOT / "data/peptide_h3_temporal_split_v3/audit.json").read_text())
record = audit["records"]["sealed_final"][0]
print(f"Testing record: {record['id']}", flush=True)

lmdb_record = read_record(Path(ROOT / "data/peptide_h3_temporal_split_v3/sealed_final.lmdb"), record["id"])
outdir = Path(ROOT / "results/publication/_debug_single")
outdir.mkdir(parents=True, exist_ok=True)
deposited = outdir / "deposited.pdb"
manifest = write_record_backbone(lmdb_record, deposited, include_antigen=True)
print(f"Manifest: {manifest}", flush=True)

ab_chains = [c for c in ("H", "L") if c in manifest]
parser = PDBParser(QUIET=True)
model = parser.get_structure("c", str(deposited))[0]
gen_config = {
    "chains": {"antibody": ab_chains, "peptide": "P"},
    "sampling": config["sampling"],
    "torsion_perturbation": config["torsion_perturbation"],
}
print(f"Starting generate_t2_1...", flush=True)
result = generate_t2_1(deposited, outdir / "t2.1", gen_config, record["id"], model, ab_chains, "P")
print(json.dumps({"torsion_hit": result["torsion_hit_tier"], "n_arms": len(result.get("arms", {}))}, indent=2))
