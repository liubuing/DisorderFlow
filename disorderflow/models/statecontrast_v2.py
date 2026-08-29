"""StateContrast-v2 model extension without modifying the frozen BFN core."""

from __future__ import annotations

import torch
import torch.nn as nn

from disorderflow.modules.statecontrast_v2 import (
    grouped_source_constrained_pose_weights,
    grouped_state_contrast_loss,
    grouped_teacher_consistency_loss,
)
from disorderflow.modules.bfn.core import sequence_cross_entropy_20

from ._base import register_model
from .bfn_model import AntibodyBFN


def masked_mean(values, mask):
    mask = mask.to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class StateContrastV2Objective(nn.Module):
    """Combine state contrast, pose regularization, and frozen-teacher order."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, state_scores, pose_quality_logits, batch):
        required = (
            "state_group_id", "state_type", "pose_source_id", "state_sample_weight",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"StateContrast-v2 batch is missing: {missing}")
        group_ids = batch["state_group_id"].long()
        state_types = batch["state_type"].long()
        # Normalize pose mass separately for target, apo, and off-target states.
        aggregation_groups = group_ids * 3 + state_types
        pose_weights, pose_kl = grouped_source_constrained_pose_weights(
            pose_quality_logits,
            aggregation_groups,
            batch["pose_source_id"].long(),
            valid=batch.get("state_valid"),
            minimum_source_mass=float(self.cfg.get("minimum_source_mass", 0.10)),
            maximum_pose_weight=float(self.cfg.get("maximum_pose_weight", 0.60)),
            prior_weights=batch.get("pose_prior_weight"),
        )
        contrast, metrics = grouped_state_contrast_loss(
            state_scores,
            group_ids,
            state_types,
            pose_weights,
            sample_weights=batch["state_sample_weight"],
            apo_margin=float(self.cfg.get("apo_margin", 0.10)),
            off_target_margin=float(self.cfg.get("off_target_margin", 0.20)),
            off_target_temperature=float(self.cfg.get("off_target_temperature", 0.25)),
        )
        losses = {
            "state_contrast_v2": contrast,
            "pose_weight_kl": pose_kl,
            "state_target_gap": metrics["mean_target_gap"],
        }
        teacher = batch.get("independent_teacher_score")
        if teacher is not None:
            losses["independent_consistency"] = grouped_teacher_consistency_loss(
                state_scores,
                teacher,
                batch.get("teacher_group_id", group_ids),
                valid=batch.get("independent_teacher_valid"),
                tie_epsilon=float(self.cfg.get("teacher_tie_epsilon", 1e-4)),
                temperature=float(self.cfg.get("teacher_temperature", 0.25)),
            )
        return losses, pose_weights


@register_model("antibody_bfn_statecontrast_v2")
class AntibodyBFNStateContrastV2(AntibodyBFN):
    """BFN plus an explicit target/apo/off-target state-scoring head."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dim = int(cfg.res_feat_dim)
        hidden = int(cfg.get("statecontrast_v2", {}).get("hidden_dim", dim))
        self.state_sequence_embedding = nn.Embedding(22, dim)
        # Candidate sequence is explicit because the default encoder masks the
        # design region. Off-target states remain mandatory: target-vs-apo alone
        # could otherwise be solved by detecting antigen presence.
        input_dim = dim * 5
        self.pose_quality_head = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        self.statecontrast_objective = StateContrastV2Objective(
            cfg.get("statecontrast_v2", {}))

    def _state_features(self, batch, res_feat):
        mask = batch["mask"].bool()
        design = batch.get("design_region_flag", batch["generate_flag"]).bool() & mask
        antigen = self.bfn._mask_antigen(batch, design)
        if antigen is None:
            antigen = torch.zeros_like(design)
        design_pool = masked_mean(res_feat, design)
        antigen_pool = masked_mean(res_feat, antigen)
        sequence_features = self.state_sequence_embedding(
            batch["aa"].clamp(min=0, max=21))
        sequence_pool = masked_mean(sequence_features, design)
        antigen_sequence_pool = masked_mean(sequence_features, antigen)
        return torch.cat(
            [design_pool, sequence_pool, antigen_pool, antigen_sequence_pool,
             sequence_pool * antigen_sequence_pool],
            dim=-1,
        )

    @staticmethod
    def _sequence_scores(logits, batch):
        """Candidate compatibility as negative mean H3 sequence NLL."""
        targets = batch["aa"].clone()
        targets[targets >= 20] = -100
        design = batch.get("design_region_flag", batch["generate_flag"]).bool()
        ce = sequence_cross_entropy_20(logits, targets)
        return -(ce * design).sum(dim=1) / design.sum(dim=1).clamp(min=1)

    def forward(self, batch):
        # Preserve the original generative/confidence losses, then add the new
        # state objective from encoder features produced by that same forward.
        receiver_outputs = []
        hook = self.bfn.receiver.register_forward_hook(
            lambda _module, _inputs, output: receiver_outputs.append(output[0]))
        try:
            losses = super().forward(batch)
        finally:
            hook.remove()
        factual_recycles = int(self.cfg.get("loss_weight", {}).get(
            "train_recycles", 1))
        if len(receiver_outputs) < factual_recycles:
            raise RuntimeError("Receiver did not produce the factual sequence logits")
        factual_logits = receiver_outputs[factual_recycles - 1]
        state_scores = self._sequence_scores(factual_logits, batch)
        # The inherited sequence loss averages every state. That would reward
        # the candidate under apo/off-target contexts while the contrastive loss
        # simultaneously pushes those scores down. Supervise sequence recovery
        # only on target-state records; negatives participate through state loss.
        targets = batch["aa"].clone()
        targets[targets >= 20] = -100
        design = batch.get("design_region_flag", batch["generate_flag"]).bool()
        target_design = design & (batch["state_type"] == 0).unsqueeze(-1)
        target_ce = sequence_cross_entropy_20(factual_logits, targets)
        losses["seq"] = (
            (target_ce * target_design).sum()
            / target_design.sum().clamp(min=1)
        )
        features = self._state_features(batch, batch["res_feat"])
        pose_logits = self.pose_quality_head(features).squeeze(-1)
        state_losses, pose_weights = self.statecontrast_objective(
            state_scores, pose_logits, batch)
        losses.update(state_losses)
        if batch.get("return_per_sample_metrics", False):
            losses["state_score_per_sample"] = state_scores.detach()
            losses["pose_weight_per_sample"] = pose_weights.detach()
        return losses

    @torch.no_grad()
    def score_states(self, batch):
        """Return fixed-sequence state and pose-quality scores for evaluation."""
        res_feat, pair_feat = self.encode(
            batch, remove_structure=False, remove_sequence=False)
        batch["pair_feat"] = pair_feat
        batch["res_feat"] = res_feat
        captured = []
        hook = self.bfn.receiver.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0]))
        try:
            self.bfn.score_fixed(batch, fixed_t=0.5)
        finally:
            hook.remove()
        if not captured:
            raise RuntimeError("Receiver did not produce fixed-sequence logits")
        features = self._state_features(batch, res_feat)
        return {
            "state_score": self._sequence_scores(captured[-1], batch),
            "pose_quality_logit": self.pose_quality_head(features).squeeze(-1),
        }
