"""Structural StateContrast dataset backed by authoritative parent coordinates."""

import copy
import gzip
import json
import os
import pickle

import lmdb
import torch
from Bio.Align import PairwiseAligner
from torch.utils.data import Dataset

from disorderflow.utils.protein.constants import CDR, ressymb_to_resindex

from ._base import register_dataset


CDR_ENUM = {
    "H1": CDR.H1, "H2": CDR.H2, "H3": CDR.H3,
    "L1": CDR.L1, "L2": CDR.L2, "L3": CDR.L3,
}
AA_SYMBOLS = "ACDEFGHIKLMNPQRSTVWYX"


def tensor_sequence(values):
    return "".join(AA_SYMBOLS[int(value)] for value in values)


def exact_alignment_map(official_sequence, parsed_sequence):
    """Map exact residues from an end-gap-free global alignment."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -2.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    alignment = aligner.align(official_sequence, parsed_sequence)[0]
    mapping = {}
    for official_block, parsed_block in zip(*alignment.aligned, strict=True):
        official_start, official_end = map(int, official_block)
        parsed_start, parsed_end = map(int, parsed_block)
        if official_end - official_start != parsed_end - parsed_start:
            continue
        for offset in range(official_end - official_start):
            official_index = official_start + offset
            parsed_index = parsed_start + offset
            if official_sequence[official_index] == parsed_sequence[parsed_index]:
                mapping[official_index] = parsed_index
    return mapping


def assign_official_imgt_cdrs(chain_data, official_sequence, cdrs, cdr_spans):
    parsed_sequence = tensor_sequence(chain_data["aa"])
    mapping = exact_alignment_map(official_sequence, parsed_sequence)
    cdr_flag = torch.zeros_like(chain_data["aa"])
    parsed_indices = {}
    for cdr_name, cdr_sequence in cdrs.items():
        span = cdr_spans[cdr_name]
        official_indices = list(range(span["start"], span["end"]))
        if official_sequence[span["start"]:span["end"]] != cdr_sequence:
            raise ValueError(f"Official {cdr_name} span does not reconstruct its sequence")
        if any(index not in mapping for index in official_indices):
            raise ValueError(f"Official {cdr_name} is not fully resolved in coordinates")
        indices = [mapping[index] for index in official_indices]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"Official {cdr_name} coordinate mapping is not contiguous")
        if parsed_sequence[indices[0]:indices[-1] + 1] != cdr_sequence:
            raise ValueError(f"Official {cdr_name} coordinate residues do not match")
        if cdr_flag[indices].any():
            raise ValueError("Official CDR coordinate mappings overlap")
        cdr_flag[indices] = int(CDR_ENUM[cdr_name])
        parsed_indices[cdr_name] = indices
    chain_data["cdr_flag"] = cdr_flag
    return parsed_indices


def require_full_antigen_mapping(antigen_data, official_sequence):
    parsed_sequence = tensor_sequence(antigen_data["aa"])
    mapping = exact_alignment_map(official_sequence, parsed_sequence)
    if len(mapping) != len(official_sequence) or len(parsed_sequence) != len(official_sequence):
        raise ValueError("Resolved antigen sequence is not fully represented by coordinates")
    indices = [mapping[index] for index in range(len(official_sequence))]
    if indices != list(range(len(parsed_sequence))) or parsed_sequence != official_sequence:
        raise ValueError("Resolved antigen coordinate sequence mismatch")
    return indices


def load_factory_records(records_dir, split, include_operations=None):
    records_dir = os.fspath(records_dir)
    manifest = json.loads(open(os.path.join(records_dir, "manifest.json"), encoding="utf-8").read())
    records = []
    allowed = set(include_operations) if include_operations else None
    for shard in manifest["shards"]:
        with gzip.open(os.path.join(records_dir, shard["path"]), "rt", encoding="ascii") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split:
                    continue
                operation = record["lineage"]["operation"]
                if allowed is not None and operation not in allowed:
                    continue
                records.append(record)
    return manifest, records


class _ParentStore:
    def __init__(self, path):
        self.path = os.fspath(path)
        self._env = None

    def _connect(self):
        if self._env is None:
            self._env = lmdb.open(
                self.path, subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False)
        return self._env

    def get(self, key):
        with self._connect().begin() as txn:
            payload = txn.get(str(key).encode())
        return None if payload is None else pickle.loads(payload)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


@register_dataset("statecontrast_structural")
class StateContrastStructuralDataset(Dataset):
    requires_complete_groups = True

    def __init__(self, cfg, transform=None):
        self.transform = transform
        self.parent_store = _ParentStore(cfg.parent_lmdb_path)
        manifest, records = load_factory_records(
            cfg.records_dir, cfg.split, cfg.get("include_operations"))
        expected_schema = cfg.get("expected_schema", "statecontrast_factory_v2_1")
        if manifest["schema_version"] != expected_schema:
            raise ValueError(
                f"Factory schema {manifest['schema_version']} does not match {expected_schema}")

        groups = {}
        for record in records:
            groups.setdefault(record["group_id"], []).append(record)
        max_groups = cfg.get("max_groups")
        if max_groups is not None and int(max_groups) < 1:
            raise ValueError("max_groups must be at least 1")
        retained = []
        group_indices = []
        scaffold_ids = []
        for group_id in sorted(groups):
            if max_groups is not None and len(group_indices) >= int(max_groups):
                break
            group = groups[group_id]
            parents = [item for item in group if item["lineage"]["operation"] == "observed_bound"]
            if len(parents) != 1:
                raise ValueError(f"Group {group_id} does not have exactly one parent")
            parent_key = parents[0]["provenance"]["source_record_id"]
            if self.parent_store.get(parent_key) is None:
                continue
            start = len(retained)
            retained.extend(group)
            indices = tuple(range(start, len(retained)))
            group_number = len(group_indices)
            group_indices.append(indices)
            scaffold_ids.extend([group_number] * len(indices))
        if not retained:
            raise ValueError("No complete StateContrast structural groups are available")
        self.records = retained
        self.group_indices = tuple(group_indices)
        self._scaffold_ids = scaffold_ids
        self.ids = [record["record_id"] for record in retained]

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _replace_sequence(chain, indices, sequence):
        if len(indices) != len(sequence):
            raise ValueError("Counterfactual sequence length does not match coordinate mapping")
        encoded = torch.tensor([ressymb_to_resindex[aa] for aa in sequence], dtype=torch.long)
        indices = torch.tensor(indices, dtype=torch.long)
        changed = chain["aa"][indices] != encoded
        chain["aa"][indices] = encoded
        changed_indices = indices[changed]
        if changed_indices.numel():
            chain["pos_heavyatom"][changed_indices, 4:] = 0
            chain["mask_heavyatom"][changed_indices, 4:] = False
            chain["torsion"][changed_indices] = 0
            chain["mask_torsion"][changed_indices] = False

    def __getitem__(self, index):
        record = self.records[index]
        parent_key = record["provenance"]["source_record_id"]
        structure = self.parent_store.get(parent_key)
        if structure is None:
            raise KeyError(f"Missing structural parent: {parent_key}")
        structure = copy.deepcopy(structure)
        mapping = structure["factory_mapping"]
        for cdr_name, sequence in record["antibody"]["cdrs"].items():
            chain_name = "heavy" if cdr_name.startswith("H") else "light"
            self._replace_sequence(
                structure[chain_name], mapping["cdr_indices"][cdr_name], sequence)
        self._replace_sequence(
            structure["antigen"], mapping["antigen_indices"], record["antigen"]["sequence"])
        if self.transform is not None:
            structure = self.transform(structure)
        group_number = self._scaffold_ids[index]
        structure.update({
            "contrastive_group_id": torch.tensor(group_number, dtype=torch.long),
            "contrastive_rank": torch.tensor(
                record["targets"]["contrastive_rank"], dtype=torch.float32),
            "contrastive_weight": torch.tensor(
                record["targets"]["training_weight"], dtype=torch.float32),
        })
        return structure

    def __del__(self):
        if hasattr(self, "parent_store"):
            self.parent_store.close()
