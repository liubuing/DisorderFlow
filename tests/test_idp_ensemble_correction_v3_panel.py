import json

import pytest

from scripts.prepare_idp_ensemble_correction_v3_panel import prepare


def test_panel_builder_blocks_unadmitted_cohort(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        """
cohort:
  minimum_exact_new_structures: 12
  minimum_independent_antibody_lineage_components: 12
  minimum_targets: 3
comparison:
  paired_arms: [ensemble, single_state]
  independent_generation_seeds: [12001, 12011, 12021, 12031]
  candidates_per_component_arm: 4
  native_controls_per_component: 1
mutation_space:
  mutable_regions: [heavy_chain_h3]
  substitution_buckets: [2, 4, 6, 8]
  maximum_h3_substitutions: 8
  candidate_count_per_bucket: 1
""",
        encoding="ascii",
    )
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status": "untouched_cohort_blocked"}), encoding="ascii")
    with pytest.raises(RuntimeError, match="has not passed"):
        prepare(config, admission, tmp_path / "plan.json")


def test_panel_builder_freezes_required_budget(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        """
cohort:
  minimum_exact_new_structures: 12
  minimum_independent_antibody_lineage_components: 12
  minimum_targets: 3
comparison:
  paired_arms: [ensemble, single_state]
  independent_generation_seeds: [12001, 12011, 12021, 12031]
  candidates_per_component_arm: 4
  native_controls_per_component: 1
mutation_space:
  mutable_regions: [heavy_chain_h3]
  substitution_buckets: [2, 4, 6, 8]
  maximum_h3_substitutions: 8
  candidate_count_per_bucket: 1
""",
        encoding="ascii",
    )
    components = [
        {
            "component_id": f"C{i:02d}",
            "target": "tau" if i < 4 else "amyloid_beta" if i < 8 else "alpha_synuclein",
            "antibody_lineage_cluster": f"L{i:02d}",
        }
        for i in range(12)
    ]
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps({
            "status": "untouched_cohort_admitted",
            "observed": {
                "exact_new_structures": 12,
                "independent_lineage_components": 12,
                "targets": 3,
            },
            "components": components,
        }),
        encoding="ascii",
    )
    plan = prepare(config, admission, tmp_path / "plan.json")
    assert plan["mutation_space"]["substitution_buckets"] == [2, 4, 6, 8]
    assert plan["mutation_space"]["maximum_h3_substitutions"] == 8
