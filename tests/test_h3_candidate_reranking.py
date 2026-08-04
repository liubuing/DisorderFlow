from scripts.benchmark_h3_candidate_reranking import (
    normalized_native_rank,
    parse_mpnn_fasta,
    unique_candidates,
)


def test_normalized_native_rank_has_expected_endpoints():
    assert normalized_native_rank(0.0, [1.0, 2.0]) == 1.0
    assert normalized_native_rank(3.0, [1.0, 2.0]) == 0.0
    assert normalized_native_rank(2.0, [1.0, 3.0]) == 0.5


def test_mpnn_parser_extracts_only_h3(tmp_path):
    fasta = tmp_path / "sample.fa"
    fasta.write_text(
        ">native\nAAAAAAAA\n"
        ">T=0.1, sample=1, score=1.25\nAACDEFGA\n",
        encoding="ascii",
    )
    rows = parse_mpnn_fasta(fasta, [2, 3, 4], 8)
    assert rows[0]["sequence"] == "CDE"
    assert rows[0]["generator_score"] == 1.25


def test_candidate_deduplication_preserves_native_generation_flag():
    rows = [
        {"sequence": "ACD"}, {"sequence": "AAA"}, {"sequence": "AAA"},
        {"sequence": "A?A"},
    ]
    unique, native_generated, failures = unique_candidates(rows, "ACD", 3)
    assert [row["sequence"] for row in unique] == ["AAA"]
    assert native_generated is True
    assert failures == 1
