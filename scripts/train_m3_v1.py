#!/usr/bin/env python3
"""Minimal single-process training loop for the M3 antibody inverse-folding model.

Uses the vendored ProteinMPNN model/featurizer but bypasses the Windows-
unfriendly ProcessPoolExecutor/DataLoader path. One chain per complex is masked
and the rest are visible context; loss is the masked-residue NLL.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ProteinMPNN" / "training"))
from model_utils import ProteinMPNN, featurize, loss_nll  # noqa: E402


def load_dataset(data_dir: Path, max_length: int = 500, excluded=None):
    rows = list(csv.DictReader((data_dir / "list.csv").open(encoding="utf-8")))
    by_pdb = {}
    for row in rows:
        pdb = row["CHAINID"].split("_")[0]
        by_pdb.setdefault(pdb, []).append(row)
    examples = []
    for pdb, chains in by_pdb.items():
        if excluded and pdb in excluded:
            continue
        folder = data_dir / "pdb" / pdb[1:3]
        meta = torch.load(folder / f"{pdb}.pt", weights_only=False)
        chain_data = {}
        for row in chains:
            chain = row["CHAINID"].split("_", 1)[1]
            chain_data[chain] = torch.load(
                folder / f"{pdb}_{chain}.pt", weights_only=False)
        chain_data = {
            name: arrays for name, arrays in chain_data.items()
            if len(arrays["seq"]) <= max_length
        }
        if not chain_data:
            continue
        names = sorted(chain_data)
        for masked in names:
            visible = [name for name in names if name != masked]
            record = {
                "name": f"{pdb}_{masked}",
                "masked_list": [masked],
                "visible_list": visible,
                "num_of_chains": len(names),
                "seq": "".join(chain_data[name]["seq"] for name in [masked] + visible),
            }
            for name in names:
                record[f"seq_chain_{name}"] = chain_data[name]["seq"]
                xyz = chain_data[name]["xyz"]
                record[f"coords_chain_{name}"] = {
                    f"N_chain_{name}": xyz[:, 0, :].numpy(),
                    f"CA_chain_{name}": xyz[:, 1, :].numpy(),
                    f"C_chain_{name}": xyz[:, 2, :].numpy(),
                    f"O_chain_{name}": xyz[:, 3, :].numpy(),
                }
            examples.append(record)
    return examples


def batch_featurize(examples, device):
    batch = []
    for example in examples:
        record = {
            "name": example["name"],
            "masked_list": example["masked_list"],
            "visible_list": example["visible_list"],
            "num_of_chains": example["num_of_chains"],
            "seq": example["seq"],
        }
        for chain in example["masked_list"] + example["visible_list"]:
            record[f"seq_chain_{chain}"] = example[f"seq_chain_{chain}"]
            record[f"coords_chain_{chain}"] = example[f"coords_chain_{chain}"]
        batch.append(record)
    return featurize(batch, device)


def evaluate(model, examples, device, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, min(len(examples), max_batches * 2), 2):
            batch = examples[start:start + 2]
            if not batch:
                continue
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = (
                batch_featurize(batch, device))
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            mask_for_loss = mask * chain_M
            loss, _loss_av, _tf = loss_nll(S, log_probs, mask_for_loss)
            weight = mask_for_loss.sum().item()
            total_loss += float(loss.sum().item())
            total_weight += weight
    model.train()
    return total_loss / max(total_weight, 1.0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "controlled_replication_v1" / "m3_dataset")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results" / "ecls_cross_arch_v1" / "m3_pilot")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=48)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--exclude-pdbs", type=Path, default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = args.data if args.data.is_absolute() else ROOT / args.data
    out = args.out if args.out.is_absolute() else ROOT / args.out
    (out / "model_weights").mkdir(parents=True, exist_ok=True)

    excluded = set()
    if args.exclude_pdbs:
        exclusion_path = args.exclude_pdbs if args.exclude_pdbs.is_absolute() else ROOT / args.exclude_pdbs
        exclusion_data = json.loads(exclusion_path.read_text(encoding="utf-8"))
        excluded = set(exclusion_data.get("excluded", {}))
        print(f"excluding {len(excluded)} PDBs", flush=True)
    examples = load_dataset(
        data_dir, max_length=args.max_length, excluded=excluded)
    random.shuffle(examples)
    if args.max_examples:
        examples = examples[:args.max_examples]
    split = max(1, int(0.9 * len(examples)))
    train_examples, valid_examples = examples[:split], examples[split:]
    print(f"examples train/valid: {len(train_examples)}/{len(valid_examples)}", flush=True)

    model = ProteinMPNN(
        node_features=args.hidden, edge_features=args.hidden,
        hidden_dim=args.hidden, num_encoder_layers=args.layers,
        num_decoder_layers=args.layers, k_neighbors=args.neighbors,
        dropout=0.1, augment_eps=0.2).to(device)
    if args.init_checkpoint:
        init_path = args.init_checkpoint if args.init_checkpoint.is_absolute() else ROOT / args.init_checkpoint
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"initialized from {init_path}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = []
    step = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        random.shuffle(train_examples)
        epoch_loss = 0.0
        epoch_weight = 0.0
        start_time = time.time()
        for start in range(0, len(train_examples), args.batch_size):
            batch = train_examples[start:start + args.batch_size]
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = (
                batch_featurize(batch, device))
            log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            mask_for_loss = mask * chain_M
            loss, _loss_av, _tf = loss_nll(S, log_probs, mask_for_loss)
            optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            weight = mask_for_loss.sum().item()
            epoch_loss += float(loss.sum().item())
            epoch_weight += weight
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {epoch_loss / max(epoch_weight, 1):.4f}",
                      flush=True)
        valid_loss = evaluate(model, valid_examples, device)
        train_loss = epoch_loss / max(epoch_weight, 1.0)
        log.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
                    "seconds": round(time.time() - start_time, 1)})
        print(json.dumps(log[-1]), flush=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "noise_level": 0.2,
            "num_edges": args.neighbors,
            "args": {key: str(value) for key, value in vars(args).items()},
        }
        torch.save(checkpoint, out / "model_weights" / "m3.pt")
        if args.save_every_epoch:
            torch.save(checkpoint, out / "model_weights" / f"epoch_{epoch}.pt")
    (out / "training_log.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'model_weights' / 'm3.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
