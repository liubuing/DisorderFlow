import sys, json, yaml, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_h3_epitope_delta import read_record, write_record_backbone
from generate_h3_peptide_t21 import generate_t2_1
from Bio.PDB import PDBParser

config = yaml.safe_load(Path(ROOT / "configs/benchmarks/peptide_h3_t2.1_temporal_final_v1.yml").read_text())
audit = json.loads(Path(ROOT / "data/peptide_h3_temporal_split_v3/audit.json").read_text())
record = audit["records"]["sealed_final"][0]
record_id = record["id"]
print(f"Record: {record_id}", flush=True)

try:
    lmdb_record = read_record(Path(ROOT / "data/peptide_h3_temporal_split_v3/sealed_final.lmdb"), record_id)
except Exception as e:
    print(f"LMDB ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

outdir = Path(ROOT / "results/publication/_debug_v2") / record_id
outdir.mkdir(parents=True, exist_ok=True)
deposited = outdir / "deposited.pdb"

try:
    manifest = write_record_backbone(lmdb_record, deposited, include_antigen=True)
except Exception as e:
    print(f"WRITE PDB ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print(f"Manifest: {manifest}", flush=True)
ab_chains = [c for c in ("H", "L") if c in manifest]
parser = PDBParser(QUIET=True)
model = parser.get_structure("c", str(deposited))[0]

gen_config = {
    "chains": {"antibody": ab_chains, "peptide": "P"},
    "sampling": config["sampling"],
    "torsion_perturbation": config["torsion_perturbation"],
}

try:
    result = generate_t2_1(deposited, outdir / "t2.1", gen_config, record_id, model, ab_chains, "P")
    (outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(f"DONE: torsion_hit={result['torsion_hit_tier']}", flush=True)
except Exception as e:
    print(f"GENERATE ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
