from scripts.build_h3_publication_package import (
    PACKAGE_FILES,
    ROOT,
    main_results,
    verify_temporal_final,
)


def test_all_lightweight_package_inputs_exist():
    assert all((ROOT / relative).exists() for relative in PACKAGE_FILES)


def test_temporal_final_hash_and_terminal_contract_are_valid():
    observed = verify_temporal_final()
    assert observed["decision"] == "accepted"
    assert observed["rerun_permitted"] is False


def test_main_result_table_preserves_positive_and_negative_conclusions():
    rows = main_results()
    interpretations = {row["interpretation"] for row in rows}
    assert "positive one-time temporal final" in interpretations
    assert "development gate rejected" in interpretations
