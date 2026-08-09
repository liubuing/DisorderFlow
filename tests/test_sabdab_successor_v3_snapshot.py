import csv
import io
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.build.acquire_sabdab_successor_v3_snapshot import acquire_snapshot
from scripts.build.discover_sabdab_successor_v3_snapshot import discover, normalize_pdb


FIELDS = [
    "INSTANCE", "PDB", "SABDAB_ID", "Hchain", "Lchain", "antigen_chain",
    "antigen_type", "antigen_name", "SABDABdepo_date", "SABDABupdate_date",
    "resolution",
]


def summary_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def row(instance, pdb, antigen_type="PEPTIDE", sabdab_id="sabdab-x"):
    return {
        "INSTANCE": instance, "PDB": pdb, "SABDAB_ID": sabdab_id,
        "Hchain": "H", "Lchain": "L", "antigen_chain": "A",
        "antigen_type": antigen_type, "antigen_name": "test antigen",
        "SABDABdepo_date": "20260801", "SABDABupdate_date": "20260802",
        "resolution": "2.0",
    }


class Response:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.content


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_archive(path, pdbs):
    content = "PDB_ID\n" + "".join(f"pdb_{pdb}\n" for pdb in pdbs)
    encoded = content.encode()
    info = tarfile.TarInfo("splits_final/abag_split.csv")
    info.size = len(encoded)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(encoded))
    return path


def test_normalize_pdb_handles_sabdab_forms():
    assert normalize_pdb("pdb_000012QA") == "12qa"
    assert normalize_pdb("pdb_00007st8_H_L") == "7st8"
    assert normalize_pdb(" 7ST8 ") == "7st8"


def test_acquisition_records_provenance_and_refuses_overwrite(tmp_path):
    content = summary_bytes([
        row("pdb_00001abc-H-L", "pdb_00001abc"),
        row("pdb_00001abc-H-L", "pdb_00001abc", antigen_type="ION"),
    ])
    retrieved = datetime(2026, 8, 7, 12, tzinfo=timezone.utc)
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(content)

    output = tmp_path / "snapshot"
    manifest = acquire_snapshot(output, opener=opener, now=lambda: retrieved)
    assert manifest["classification"] == "metadata_only"
    assert manifest["row_count"] == 2
    assert manifest["instance_count"] == manifest["pdb_count"] == 1
    assert manifest["duplicate_instance_rows"] == 1
    assert manifest["byte_count"] == len(content)
    assert manifest["minimum_update_date"] == manifest["maximum_update_date"] == "20260802"
    assert calls[0][0].endswith("/api/download/all-summary")

    with pytest.raises(FileExistsError):
        acquire_snapshot(output, opener=lambda *args, **kwargs: pytest.fail("network called"))


def test_exact_diff_excludes_every_exposure_and_preserves_instances(tmp_path):
    old = row("pdb_00001old-H-L", "pdb_00001old")
    rows = [
        old,
        row("pdb_00001old-A-B", "pdb_00001old"),
        row("pdb_00001arc-H-L", "pdb_00001arc"),
        row("pdb_00001aud-H-L", "pdb_00001aud"),
        row("pdb_00001v1x-H-L", "pdb_00001v1x"),
        row("pdb_00001v2x-H-L", "pdb_00001v2x"),
        row("pdb_00001v3x-H-L", "pdb_00001v3x"),
        row("pdb_00003stb-H-L", "pdb_00003stb"),
        row("pdb_00001non-H-L", "pdb_00001non", "PROTEIN"),
        row("pdb_00001new-H-L", "pdb_00001new", sabdab_id="shared"),
        row("pdb_00001new-A-B", "pdb_00001new", sabdab_id="shared"),
    ]
    snapshot = tmp_path / "snapshot"
    acquisition_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    acquire_snapshot(
        snapshot, opener=lambda *args, **kwargs: Response(summary_bytes(rows)),
        now=lambda: acquisition_time,
    )
    baseline = tmp_path / "baseline.csv"
    baseline.write_bytes(summary_bytes([old]))
    archive = write_archive(tmp_path / "splits.tar.gz", ["1arc"])
    audit = write_json(tmp_path / "audit.json", {
        "records": {"sealed": [{"id": "pdb_00001aud_H_L", "axis_values": {"pdb_id": ["1aud"]}}]}
    })
    v1_candidates = [
        {"instance": f"pdb_{index:08x}-H-L", "pdb_id": f"{index:04x}"}
        for index in range(111)
    ]
    v1_candidates[0] = {"instance": "pdb_00001v1x-H-L", "pdb_id": "1v1x"}
    v1 = write_json(tmp_path / "v1.json", {"candidates": v1_candidates})
    v2 = write_json(tmp_path / "v2.json", {
        "n_reference_independent_records": 1,
        "components": [{"members": ["pdb_00001v2x-H-L"]}],
    })
    v3_records = [{"id": f"pdb_{index + 500:08x}_H_L"} for index in range(258)]
    v3_records[0] = {"id": "pdb_00001v3x_H_L"}
    v3 = write_json(tmp_path / "v3.json", {"records": v3_records})
    output = tmp_path / "discovery.json"

    report = discover(
        snapshot, output, baseline, archive, [audit], v1, v2, v3,
        now=lambda: datetime.now(timezone.utc),
    )
    assert [item["instance"] for item in report["candidates"]] == [
        "pdb_00001new-A-B", "pdb_00001new-H-L",
    ]
    assert report["counts"]["candidate_instances"] == 2
    assert report["counts"]["candidate_pdbs"] == 1
    assert all(value["sha256"] for value in report["inputs"].values())
    assert all(report["exclusion_match_counts"][reason] for reason in (
        "baseline_instance", "baseline_pdb", "historical_archive_pdb",
        "peptide_h3_audit_pdb", "v1_discovery_pdb", "v2_holdout_pdb",
        "v3_development_pdb", "fixed_pdb", "not_peptide",
    ))

    with pytest.raises(FileExistsError):
        discover(snapshot, output, baseline, archive, [audit], v1, v2, v3)
