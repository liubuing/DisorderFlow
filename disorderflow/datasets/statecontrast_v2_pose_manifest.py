"""Direct PDB dataset for explicit StateContrast-v2 pose manifests."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import torch
from Bio.PDB import PDBParser
from torch.utils.data import Dataset

from disorderflow.utils.protein import parsers
from disorderflow.utils.protein.constants import CDR, ressymb_to_resindex

from ._base import register_dataset
from .custom import _label_heavy_chain_cdr, _label_light_chain_cdr
from .statecontrast_v2_structural import STATE_TYPES

ROOT = Path(__file__).resolve().parents[2]


def _merge_chains(chains):
    if not chains:
        return None
    if len(chains) == 1:
        return chains[0]
    output = {}
    for key in chains[0]:
        values = [chain[key] for chain in chains]
        output[key] = (
            torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor)
            else sum(values, start=[]) if isinstance(values[0], list)
            else values[0]
        )
    return output


def _sequence(data):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    return "".join(alphabet[int(value)] for value in data["aa"])


@lru_cache(maxsize=256)
def _parse_pose(path_text, antibody_tuple, antigen_tuple):
    path = Path(path_text)
    model = PDBParser(QUIET=True).get_structure(path.stem, str(path))[0]
    antibody = list(antibody_tuple)
    antigens = list(antigen_tuple)
    heavy, heavy_map = parsers.parse_biopython_structure(model[antibody[0]])
    heavy, heavy_map = _label_heavy_chain_cdr(heavy, heavy_map)
    light = light_map = None
    if len(antibody) > 1:
        light, light_map = parsers.parse_biopython_structure(model[antibody[1]])
        light, light_map = _label_light_chain_cdr(light, light_map)
    antigen_parts = [parsers.parse_biopython_structure(model[chain])[0] for chain in antigens]
    antigen = _merge_chains(antigen_parts)
    return {
        "heavy": heavy,
        "heavy_seqmap": heavy_map,
        "light": light,
        "light_seqmap": light_map,
        "antigen": antigen,
        "antigen_seqmap": None,
    }


def parse_record_structure(record):
    path = ROOT / record["coordinate"]["path"]
    antibody = tuple(record["state"]["antibody_chains"])
    antigens = tuple(record["state"]["antigen_chains"])
    parsed = copy.deepcopy(_parse_pose(str(path), antibody, antigens))
    heavy = parsed["heavy"]
    light = parsed["light"]
    antigen = parsed["antigen"]
    override = record["state"].get("antigen_sequence_override")
    if override is not None:
        if antigen is None or not override:
            raise ValueError("Antigen sequence override requires antigen coordinates")
        mapped = torch.linspace(
            0, len(override) - 1, len(antigen["aa"])).round().long()
        encoded_override = torch.tensor([
            ressymb_to_resindex[override[index]] for index in mapped
        ], dtype=torch.long)
        changed_antigen = antigen["aa"] != encoded_override
        antigen["aa"] = encoded_override
        changed_positions = torch.where(changed_antigen)[0]
        if changed_positions.numel():
            antigen["pos_heavyatom"][changed_positions, 4:] = 0
            antigen["mask_heavyatom"][changed_positions, 4:] = False
            antigen["torsion"][changed_positions] = 0
            antigen["mask_torsion"][changed_positions] = False
    native = record["native_h3"]
    start = _sequence(heavy).find(native)
    if start < 0:
        raise ValueError(f"Native H3 not found in {record['record_id']}")
    positions = torch.arange(start, start + len(native))
    heavy["cdr_flag"] = torch.zeros_like(heavy["aa"])
    heavy["cdr_flag"][positions] = int(CDR.H3)
    if light is not None and "cdr_flag" not in light:
        light["cdr_flag"] = torch.zeros_like(light["aa"])
    if antigen is not None:
        antigen["cdr_flag"] = torch.zeros_like(antigen["aa"])

    encoded = torch.tensor([
        ressymb_to_resindex[aa] for aa in record["candidate_sequence"]
    ], dtype=torch.long)
    changed = heavy["aa"][positions] != encoded
    heavy["aa"][positions] = encoded
    changed_positions = positions[changed]
    if changed_positions.numel():
        heavy["pos_heavyatom"][changed_positions, 4:] = 0
        heavy["mask_heavyatom"][changed_positions, 4:] = False
        heavy["torsion"][changed_positions] = 0
        heavy["mask_torsion"][changed_positions] = False
    parsed["id"] = record["record_id"]
    return parsed


@register_dataset("statecontrast_v2_pose_manifest")
class StateContrastV2PoseManifestDataset(Dataset):
    """Load verified explicit-state PDBs without a legacy parent LMDB."""

    requires_complete_groups = True

    def __init__(self, cfg, transform=None):
        manifest = json.loads(Path(cfg.manifest_path).read_text(encoding="ascii"))
        heldout_fold = cfg.get("heldout_fold")
        fold_role = cfg.get("fold_role")
        if heldout_fold is not None:
            if fold_role not in ("train", "validation"):
                raise ValueError("fold_role must be train or validation")
            records = [
                row for row in manifest["records"]
                if ((row["fold_id"] != int(heldout_fold))
                    if fold_role == "train"
                    else (row["fold_id"] == int(heldout_fold)))
            ]
        else:
            records = (
                list(manifest["records"]) if cfg.split == "all" else
                [row for row in manifest["records"] if row["split"] == cfg.split]
            )
        groups = {}
        state_group_names = sorted({row["group_id"] for row in records})
        state_group_index = {
            group_id: index for index, group_id in enumerate(state_group_names)
        }
        for record in records:
            # A teacher super-group contains the matched ensemble/single-state
            # candidates for one component and mutation bucket. Packing the
            # super-group keeps teacher pair comparisons inside one batch.
            groups.setdefault(record["teacher_group_id"], []).append(record)
        self.records = []
        self.group_indices = []
        self.group_source_ids = []
        self._state_group_number = []
        self._teacher_group_number = []
        for teacher_number, teacher_group_id in enumerate(sorted(groups)):
            group = groups[teacher_group_id]
            for candidate_group in {row["group_id"] for row in group}:
                state_types = {
                    row["state"]["type"] for row in group
                    if row["group_id"] == candidate_group
                }
                if "target" not in state_types or not ({"apo", "off_target"} & state_types):
                    raise ValueError(f"Incomplete explicit state group: {candidate_group}")
            start = len(self.records)
            self.records.extend(group)
            indices = tuple(range(start, len(self.records)))
            self.group_indices.append(indices)
            self._state_group_number.extend([
                state_group_index[row["group_id"]] for row in group
            ])
            self._teacher_group_number.extend([teacher_number] * len(indices))
            sources = {row["provenance"]["source"] for row in group}
            if len(sources) != 1:
                raise ValueError(f"Group spans dataset sources: {teacher_group_id}")
            self.group_source_ids.append(next(iter(sources)))
        self.group_indices = tuple(self.group_indices)
        self.group_source_ids = tuple(self.group_source_ids)
        self.transform = transform
        pose_sources = sorted({row["state"]["source"] for row in self.records})
        self._pose_source_index = {name: index for index, name in enumerate(pose_sources)}
        arms = sorted({row["origin_arm"] for row in self.records})
        self._arm_index = {name: index for index, name in enumerate(arms)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        structure = parse_record_structure(record)
        if self.transform is not None:
            structure = self.transform(copy.deepcopy(structure))
        state = record["state"]
        teacher = record["targets"].get("independent_teacher_score")
        structure.update({
            "contrastive_group_id": torch.tensor(self._state_group_number[index]),
            "contrastive_rank": torch.tensor(1.0 if state["type"] == "target" else 0.0),
            "contrastive_weight": torch.tensor(record["targets"]["state_training_weight"]),
            "design_region_flag": structure["generate_flag"].clone(),
            "state_group_id": torch.tensor(self._state_group_number[index]),
            "teacher_group_id": torch.tensor(self._teacher_group_number[index]),
            "state_type": torch.tensor(STATE_TYPES[state["type"]]),
            "pose_source_id": torch.tensor(self._pose_source_index[state["source"]]),
            "pose_prior_weight": torch.tensor(float(state["prior_weight"])),
            "state_sample_weight": torch.tensor(
                float(record["targets"]["state_training_weight"])),
            "state_valid": torch.tensor(bool(state.get("admitted", True))),
            "independent_teacher_score": torch.tensor(float(teacher or 0.0)),
            "independent_teacher_valid": torch.tensor(teacher is not None),
            "candidate_arm": torch.tensor(self._arm_index[record["origin_arm"]]),
            "substitution_bucket": torch.tensor(int(record["substitution_bucket"])),
        })
        return structure
