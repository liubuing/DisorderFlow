from scripts.pipeline.export_5csz_h3_seqres_constructs import sequence_sha256


def test_sequence_sha256_is_stable():
    assert sequence_sha256("ABC") == sequence_sha256("ABC")
    assert sequence_sha256("ABC") != sequence_sha256("ABD")
