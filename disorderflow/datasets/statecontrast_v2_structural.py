"""Structural StateContrast-v2 records with explicit state/source metadata."""

from __future__ import annotations

import torch

from ._base import register_dataset
from .statecontrast_structural import StateContrastStructuralDataset


STATE_TYPES = {"target": 0, "apo": 1, "off_target": 2}


@register_dataset("statecontrast_v2_structural")
class StateContrastV2StructuralDataset(StateContrastStructuralDataset):
    """Fail-closed extension requiring real state and source annotations."""

    def __init__(self, cfg, transform=None):
        super().__init__(cfg, transform=transform)
        source_names = sorted({
            str(record.get("state", {}).get("source")) for record in self.records
        })
        if "None" in source_names:
            raise ValueError("Every StateContrast-v2 record requires state.source")
        self._source_index = {name: index for index, name in enumerate(source_names)}
        self.group_source_ids = []
        for group in self.group_indices:
            # Batch balancing uses the declared dataset source of the target
            # member. Pose-source balancing remains a per-record model input.
            target_sources = {
                self.records[index]["provenance"]["source"] for index in group
                if self.records[index].get("state", {}).get("type") == "target"
            }
            if len(target_sources) != 1:
                raise ValueError("Each group requires one target dataset source")
            self.group_source_ids.append(next(iter(target_sources)))
        self.group_source_ids = tuple(self.group_source_ids)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        record = self.records[index]
        state = record.get("state", {})
        state_type = state.get("type")
        if state_type not in STATE_TYPES:
            raise ValueError(f"Unsupported or missing state.type: {state_type}")
        teacher = record.get("targets", {}).get("independent_teacher_score")
        sample.update({
            "design_region_flag": sample["generate_flag"].clone(),
            "state_group_id": sample["contrastive_group_id"].clone(),
            "state_type": torch.tensor(STATE_TYPES[state_type], dtype=torch.long),
            "pose_source_id": torch.tensor(
                self._source_index[str(state["source"])], dtype=torch.long),
            "pose_prior_weight": torch.tensor(
                float(state.get("prior_weight", 1.0)), dtype=torch.float32),
            "state_sample_weight": torch.tensor(
                float(record["targets"].get("state_training_weight", 1.0)),
                dtype=torch.float32),
            "state_valid": torch.tensor(bool(state.get("admitted", True))),
            "independent_teacher_score": torch.tensor(
                float(teacher) if teacher is not None else 0.0,
                dtype=torch.float32),
            "independent_teacher_valid": torch.tensor(teacher is not None),
        })
        return sample
