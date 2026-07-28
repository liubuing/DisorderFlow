#!/usr/bin/env python
"""Integration contract for StateContrast-Ab inside DisorderFlow workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class StateContrastWorkflowConfig:
    domain: str = "idp_antibody"
    positive_state_source: str = "contact_map"
    negative_state_source: str = "synthetic_perturbation"
    design_engine: str = "bfn_or_contact_guided"
    attribution_mode: str = "position_effect_size"
    validation_mode: str = "heldout_negative_states"
    fold_sidecheck: str = "optional_colabfold"
    required_references: int = 2
    notes: List[str] = field(default_factory=list)


def default_statecontrast_workflow() -> StateContrastWorkflowConfig:
    return StateContrastWorkflowConfig(
        notes=[
            "Use StateContrast as a ranker and position-discovery layer, not a residue-identity oracle.",
            "Keep parent-only, random-position, and shuffled-position controls in every benchmark.",
            "Treat AF2/ColabFold as fold sanity evidence, not IDP binding proof.",
        ]
    )


def workflow_stages(config: StateContrastWorkflowConfig | None = None) -> List[Dict]:
    cfg = config or default_statecontrast_workflow()
    return [
        {
            "stage": "domain_intake",
            "purpose": "Load target/scaffold/reference complexes and identify IDP/disordered regions.",
            "gpu_required": False,
        },
        {
            "stage": "candidate_generation",
            "purpose": f"Generate candidate paratopes using {cfg.design_engine}.",
            "gpu_required": cfg.design_engine.startswith("bfn"),
        },
        {
            "stage": "statecontrast_scoring",
            "purpose": "Rank candidates by positive-vs-negative state specificity-gap delta.",
            "gpu_required": False,
        },
        {
            "stage": "position_effect_attribution",
            "purpose": "Discover recurrent state-sensitive paratope positions under held-out controls.",
            "gpu_required": False,
        },
        {
            "stage": "constrained_redesign",
            "purpose": "Restrict redesign/search to validated state-sensitive positions.",
            "gpu_required": cfg.design_engine.startswith("bfn"),
        },
        {
            "stage": "fold_sidecheck",
            "purpose": "Optional fold sanity check for shortlisted constructs.",
            "gpu_required": cfg.fold_sidecheck != "none",
        },
        {
            "stage": "final_cascade",
            "purpose": "Report method evidence, controls, limitations, and experimental triage set.",
            "gpu_required": False,
        },
    ]


def readiness_requirements(config: StateContrastWorkflowConfig | None = None) -> Dict:
    stages = workflow_stages(config)
    return {
        "gpu_required_for_full_workflow": any(s["gpu_required"] for s in stages),
        "cpu_only_supported_for_statecontrast_benchmark": True,
        "required_controls": [
            "heldout_negative_states",
            "parent_only",
            "random_position",
            "shuffled_position",
            "multi_seed_replicates",
        ],
        "stages": stages,
    }


def validate_reference_count(
    references: Iterable[Dict],
    minimum: int = 2,
    require_verified: bool = False,
) -> Dict:
    refs = list(references)
    identified = [r for r in refs if r.get("path") or r.get("complex_pdb") or r.get("pdb")]
    reliable = identified
    if require_verified:
        reliable = [
            r for r in identified
            if r.get("reliability") == "reliable" and r.get("provenance_verified") is True
        ]
    return {
        "n_references": len(reliable),
        "n_declared_references": len(refs),
        "minimum": minimum,
        "require_verified": require_verified,
        "status": "pass" if len(reliable) >= minimum else "fail",
    }


def integration_claim_level(n_references: int, has_real_negative_ensembles: bool, has_wetlab: bool) -> str:
    if has_wetlab and has_real_negative_ensembles and n_references >= 8:
        return "validated_domain_workflow"
    if has_real_negative_ensembles and n_references >= 4:
        return "generalizing_computational_workflow"
    if n_references >= 2:
        return "proof_of_concept_position_discovery"
    return "exploratory_only"
