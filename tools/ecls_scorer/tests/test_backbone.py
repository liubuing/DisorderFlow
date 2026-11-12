import pytest

from ecls_scorer.backbone import locate_h3_indices, write_standardized_backbone


def _atom(serial, atom_name, resname, chain, resseq, x, y, z, element,
          altloc=" ", occupancy=1.0):
    return (f"ATOM  {serial:5d} {atom_name:<4s}{altloc}{resname:>3s} {chain:1s}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:5.2f}  0.00          {element:>2s}\n")


def _residue_atoms(serial, resname, chain, resseq, *, oxygen=True, altloc_copies=None):
    lines = []
    coords = {"N": (1.0, 0.0, 0.0), "CA": (2.0, 0.0, 0.0),
              "C": (3.0, 0.0, 0.0), "O": (4.0, 0.0, 0.0)}
    if not oxygen:
        del coords["O"]
    for atom_name, (x, y, z) in coords.items():
        element = atom_name[0]
        if atom_name == "CA" and altloc_copies:
            for altloc, occupancy in altloc_copies:
                lines.append(_atom(serial, atom_name, resname, chain, resseq,
                                   x + occupancy, y, z, element,
                                   altloc=altloc, occupancy=occupancy))
                serial += 1
        else:
            lines.append(_atom(serial, atom_name, resname, chain, resseq, x, y, z, element))
            serial += 1
    return lines, serial


def _write_pdb(path, lines):
    path.write_text("".join(lines) + "TER\nEND\n", encoding="ascii")


def _chain(lines, serial, resnames, chain="A", start=1, **kwargs):
    for offset, resname in enumerate(resnames, start):
        produced, serial = _residue_atoms(serial, resname, chain, offset, **kwargs)
        lines.extend(produced)
    return serial


def test_standardization_renumbers_and_extracts_sequences(tmp_path):
    pdb = tmp_path / "complex.pdb"
    lines = []
    serial = _chain(lines, 1, ["ALA", "GLY", "SER"], chain="A")
    serial = _chain(lines, serial, ["PHE"], chain="B")
    _write_pdb(pdb, lines)
    manifest = write_standardized_backbone(str(pdb), str(tmp_path / "std.pdb"), ["A", "B"])
    assert manifest["chains"]["A"]["sequence"] == "AGS"
    assert manifest["chains"]["A"]["length"] == 3
    assert manifest["chains"]["B"]["sequence"] == "F"
    mapping = manifest["chains"]["A"]["residue_mapping"]
    assert [row["standardized_resseq"] for row in mapping] == [1, 2, 3]
    assert [row["original_resseq"] for row in mapping] == [1, 2, 3]


def test_peptide_stripped_state_keeps_retained_chains_identical(tmp_path):
    pdb = tmp_path / "complex.pdb"
    lines = []
    serial = _chain(lines, 1, ["ALA", "GLY"], chain="H")
    serial = _chain(lines, serial, ["PHE", "TRP"], chain="P")
    _write_pdb(pdb, lines)
    complex_manifest = write_standardized_backbone(
        str(pdb), str(tmp_path / "complex_std.pdb"), ["H", "P"])
    apo_manifest = write_standardized_backbone(
        str(pdb), str(tmp_path / "apo_std.pdb"), ["H"])
    assert complex_manifest["chains"]["H"]["sequence"] == \
        apo_manifest["chains"]["H"]["sequence"] == "AG"


def test_altloc_resolved_by_occupancy(tmp_path):
    pdb = tmp_path / "complex.pdb"
    lines = []
    produced, serial = _residue_atoms(1, "ALA", "A", 1,
                                      altloc_copies=[("A", 0.3), ("B", 0.7)])
    lines.extend(produced)
    _write_pdb(pdb, lines)
    manifest = write_standardized_backbone(str(pdb), str(tmp_path / "std.pdb"), ["A"])
    assert manifest["chains"]["A"]["sequence"] == "A"
    # The 0.7-occupancy conformer carries x = 2.0 + 0.7
    text = (tmp_path / "std.pdb").read_text(encoding="ascii")
    assert "  2.700" in text


def test_missing_oxygen_raises(tmp_path):
    pdb = tmp_path / "complex.pdb"
    lines = []
    produced, serial = _residue_atoms(1, "ALA", "A", 1, oxygen=False)
    lines.extend(produced)
    _write_pdb(pdb, lines)
    with pytest.raises(ValueError, match="Incomplete backbone"):
        write_standardized_backbone(str(pdb), str(tmp_path / "std.pdb"), ["A"])


def test_hetero_water_skipped(tmp_path):
    pdb = tmp_path / "complex.pdb"
    lines = []
    serial = _chain(lines, 1, ["GLY"], chain="A")
    lines.append(_atom(serial, "O", "HOH", "A", 99, 9.0, 9.0, 9.0, "O"))
    _write_pdb(pdb, lines)
    manifest = write_standardized_backbone(str(pdb), str(tmp_path / "std.pdb"), ["A"])
    assert manifest["chains"]["A"]["sequence"] == "G"


def test_locate_h3_indices_unique_and_explicit_start():
    heavy = "VQLLESGGGLVQPGGSLRLSCAASGFTFSNYGMSWVRQAPGKGLEWVASIRSGGGRTYYSDNVKGRFTIS" \
            "RDNAKNSLYLQMNSLRAEDTALYYCVRYDHYSGSSDYWGQGTLVTVSS"
    h3 = "VRYDHYSGSSDY"
    indices = locate_h3_indices(heavy, h3)
    start = heavy.index(h3)
    assert indices == list(range(start, start + len(h3)))
    assert locate_h3_indices(heavy, h3, start + 1) == indices


def test_locate_h3_indices_rejects_ambiguous_or_missing():
    with pytest.raises(ValueError, match="not found"):
        locate_h3_indices("AAAA", "CCCC")
    with pytest.raises(ValueError, match="matches 2 positions"):
        locate_h3_indices("AAXAA", "AA")
    with pytest.raises(ValueError, match="is .* not"):
        locate_h3_indices("ABCDEF", "CDE", 1)
