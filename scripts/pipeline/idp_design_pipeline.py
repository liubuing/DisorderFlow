#!/usr/bin/env python
"""
Direction 1+3 Combined: IDP Ordered-Region Antibody Design Pipeline

Pipeline:
  1. Load IDP from V11 LMDB →extract AF2 structure as PDB
  2. BFN disorder head →per-residue disorder scores
  3. Find ordered segments (D < 0.3, len >= 8)
  4. For each ordered segment →BFN CDR design
  5. Output: designable epitopes + designed sequences + confidence scores

Usage:
  python scripts/pipeline/idp_design_pipeline.py --pdb-id P38936
  python scripts/pipeline/idp_design_pipeline.py --all-idps --max 5
"""

import os, sys, pickle, lmdb, argparse, json, tempfile, time
import torch
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "modules"))
os.chdir(_project_root)

AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_bfn_model(device):
    from disorderflow.models import get_model
    from disorderflow.utils.misc import load_config

    ckpt_path = os.path.join(
        _project_root,
        "logs/bfn_idp_phase2_xpu_2026_05_28__01_22_06/checkpoints/best.pt",
    )
    config, _ = load_config(
        os.path.join(_project_root, "configs/train/bfn_idp_phase2_cuda.yml")
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt["config"].model
    if (
        hasattr(ckpt["config"], "train")
        and hasattr(ckpt["config"].train, "loss_weights")
    ):
        mc["loss_weight"] = dict(ckpt["config"].train.loss_weights)
    model = get_model(mc).to(device).eval()
    return model, config


def batch_to_pdb(batch, output_path, chain_id="A"):
    """Extract C伪 trace PDB from LMDB batch data."""
    pos = batch["pos_heavyatom"]  # (L, 37, 3) or (L, 5, 3)
    aa = batch["aa"]
    mask = batch.get("mask_heavyatom", None)
    if mask is not None:
        ca_mask = mask[:, 1].bool() if mask.shape[1] > 1 else mask[:, 0].bool()
    else:
        ca_mask = torch.ones(pos.shape[0], dtype=torch.bool)

    with open(output_path, "w") as f:
        atom_idx = 1
        for i in range(pos.shape[0]):
            if not ca_mask[i]:
                continue
            x, y, z = pos[i, 1, :].tolist()  # C伪 is index 1
            aa_code = AA_LETTERS[aa[i].item()] if aa[i].item() < 20 else "GLY"
            f.write(
                f"ATOM  {atom_idx:5d}  CA  {aa_code:3s} {chain_id}{i+1:4d}"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_idx += 1
        f.write("TER\nEND\n")
    return output_path


def predict_disorder_regions(model, config, batch, device):
    """Run BFN disorder prediction and return ordered segments."""
    from disorderflow.utils.train import recursive_to
    from disorderflow.utils.data import PaddingCollate
    from disorderflow.utils.misc import seed_all

    # Prepare batch
    input_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            input_batch[k] = v.unsqueeze(0)

    collate = PaddingCollate()
    input_batch = collate([input_batch])
    input_batch = recursive_to(input_batch, device)

    seed_all(42)
    sample_opt = {"deterministic": True, "return_disorder": True, "num_recycles": 1}

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            traj = model.sample(input_batch, sample_opt=sample_opt)

    if "disorder" not in traj:
        raise RuntimeError("Model does not have disorder head")

    mask_valid = input_batch["mask"][0].bool()
    disorder_logits = traj["disorder"][0][mask_valid]
    disorder_scores = torch.sigmoid(disorder_logits).cpu().numpy()
    plddt = traj.get("plddt", None)
    if plddt is not None:
        plddt = plddt[0][mask_valid].cpu().numpy()

    aa = input_batch["aa"][0][mask_valid]
    sequence = "".join(
        AA_LETTERS[a.item()] if a.item() < 20 else "X" for a in aa
    )

    # Find ordered segments
    ordered = disorder_scores < 0.3
    segments = []
    seg_start = None
    for i in range(len(ordered)):
        if ordered[i] and seg_start is None:
            seg_start = i
        elif not ordered[i] and seg_start is not None:
            if i - seg_start >= 8:
                seg_d = disorder_scores[seg_start:i].mean()
                segments.append(
                    {
                        "start": seg_start + 1,
                        "end": i,
                        "length": i - seg_start,
                        "mean_disorder": float(seg_d),
                        "sequence": sequence[seg_start:i],
                    }
                )
            seg_start = None
    if seg_start is not None and len(ordered) - seg_start >= 8:
        seg_d = disorder_scores[seg_start:].mean()
        segments.append(
            {
                "start": seg_start + 1,
                "end": len(ordered),
                "length": len(ordered) - seg_start,
                "mean_disorder": float(seg_d),
                "sequence": sequence[seg_start:],
            }
        )

    segments.sort(key=lambda s: s["mean_disorder"])
    return {
        "pdb_id": batch.get("pdb_id", "?"),
        "length": len(sequence),
        "sequence": sequence,
        "mean_disorder": float(disorder_scores.mean()),
        "n_ordered": int(ordered.sum()),
        "n_disordered": int((~ordered).sum()),
        "disorder_scores": disorder_scores.tolist(),
        "plddt_pred": plddt.tolist() if plddt is not None else None,
        "ordered_segments": segments,
    }


def run_design_on_segment(pdb_path, segment, chain_id="A", device="cuda", n_samples=3):
    """Run BFN CDR design on an ordered segment."""
    from bfn_loader import run_bfn_design

    region_spec = f"{chain_id}:{segment['start']}-{segment['end']}"
    try:
        results = run_bfn_design(
            pdb_path,
            region_spec,
            num_samples=n_samples,
            stochastic=True,
            device=device,
        )
        return {"region": region_spec, "segment": segment, "designs": results}
    except Exception as e:
        return {"region": region_spec, "segment": segment, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Direction 1+3: IDP Ordered-Region Antibody Design Pipeline"
    )
    parser.add_argument("--pdb-id", type=str, default=None, help="Specific PDB ID to process")
    parser.add_argument("--all-idps", action="store_true", help="Process all suitable IDPs")
    parser.add_argument("--max", type=int, default=3, dest="max_idps")
    parser.add_argument("--n-designs", type=int, default=3, help="Designs per segment")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"Device: {device}")

    # Load model
    print("Loading BFN model...")
    model, config = load_bfn_model(device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Open V11
    v11_path = os.path.join(_project_root, "data/confidence_merged_v11/confidence_val.lmdb")
    env = lmdb.open(v11_path, readonly=True, lock=False)

    # Find IDP entries
    with env.begin() as txn:
        n_total = pickle.loads(txn.get(b"__len__"))
        idp_entries = []
        for i in range(n_total):
            entry = pickle.loads(txn.get(f"{i:08d}".encode()))
            if not entry.get("is_idp", False):
                continue
            L = len(entry.get("sequence", ""))
            disorder_f = entry.get("disorder_fraction", 0)
            if 0.15 < disorder_f < 0.85 and 100 <= L <= 300:
                idp_entries.append((i, entry))
                if args.pdb_id and entry.get("pdb_id") == args.pdb_id:
                    idp_entries = [(i, entry)]
                    break

    if args.pdb_id and not idp_entries:
        print(f"PDB ID {args.pdb_id} not found in V11 val")
        env.close()
        sys.exit(1)

    idp_entries = idp_entries[: args.max_idps]
    print(f"Processing {len(idp_entries)} IDP(s)")

    all_results = []
    out_dir = args.output or tempfile.mkdtemp(prefix="idp_design_")
    os.makedirs(out_dir, exist_ok=True)

    for idx, (src_idx, entry) in enumerate(idp_entries):
        pdb_id = entry.get("pdb_id", f"idp_{idx}")
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(idp_entries)}] {pdb_id}")
        print(f"{'='*60}")

        # Step 1: Extract PDB
        pdb_path = os.path.join(out_dir, f"{pdb_id}.pdb")
        batch_to_pdb(entry["batch"], pdb_path)
        print(f"  PDB written: {pdb_path}")

        # Step 2: Disorder prediction
        print("  Running disorder prediction...")
        disorder_result = predict_disorder_regions(model, config, entry["batch"], device)
        print(f"  Mean disorder: {disorder_result['mean_disorder']:.4f}")
        print(f"  Ordered residues: {disorder_result['n_ordered']}/{disorder_result['length']}")
        print(f"  Ordered segments found: {len(disorder_result['ordered_segments'])}")

        for seg in disorder_result["ordered_segments"][:5]:
            print(
                f"    [{seg['start']:4d}-{seg['end']:4d}] "
                f"L={seg['length']:3d}  D={seg['mean_disorder']:.3f}  "
                f"{seg['sequence'][:30]}"
            )

        # Step 3: Design on top segments
        design_results = []
        for seg in disorder_result["ordered_segments"][:2]:  # top 2 segments
            print(f"\n  Designing on segment [{seg['start']}-{seg['end']}]...")
            t0 = time.time()
            result = run_design_on_segment(
                pdb_path, seg, device=device, n_samples=args.n_designs
            )
            elapsed = time.time() - t0
            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    {len(result['designs'])} designs in {elapsed:.1f}s")
                for d in result["designs"]:
                    print(
                        f"      {d.get('sequence','?')[:40]}... "
                        f"pLDDT={d.get('plddt','?')} "
                        f"ipTM={d.get('iptm','?')}"
                    )
            design_results.append(result)

        all_results.append(
            {
                "pdb_id": pdb_id,
                "disorder_analysis": disorder_result,
                "designs": design_results,
            }
        )

    env.close()

    # Save results
    results_path = os.path.join(out_dir, "pipeline_results.json")
    # Clean non-serializable
    clean_results = []
    for r in all_results:
        cr = {
            "pdb_id": r["pdb_id"],
            "disorder_analysis": {
                k: v
                for k, v in r["disorder_analysis"].items()
                if k not in ("disorder_scores", "plddt_pred")
            },
            "ordered_segments": r["disorder_analysis"]["ordered_segments"],
            "designs": r["designs"],
        }
        clean_results.append(cr)

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")
    print(f"PDB files: {out_dir}/")


if __name__ == "__main__":
    main()
