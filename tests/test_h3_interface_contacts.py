from pathlib import Path

import pytest

from h3_interface_contacts import (
    extract_h3_interface,
    score_h3_sequence,
    serialize_interface,
    write_standardized_backbone,
)


def _atom(serial, name, resname, chain, resseq, x, y, z, icode="", element=None):
    element = element or name[0]
    return (
        f"ATOM  {serial:5d} {name:^4s} {resname:>3s} {chain:1s}"
        f"{resseq:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 20.00          {element:>2s}\n"
    )


def _write_toy_complex(path: Path):
    aa3 = {"A": "ALA", "C": "CYS", "D": "ASP", "F": "PHE", "G": "GLY",
           "K": "LYS", "S": "SER", "W": "TRP", "Y": "TYR"}
    heavy = "A" * 70 + "C" + "DYSA" + "WG" + "A" * 10
    lines = []
    serial = 1
    for index, aa in enumerate(heavy, 1):
        x = float(index * 8)
        if 72 <= index <= 75:
            x = float((index - 72) * 3)
        icode = "A" if index == 73 else ""
        for atom_name, offset in (("N", -0.5), ("CA", 0.0), ("C", 0.5), ("O", 1.0)):
            lines.append(_atom(
                serial, atom_name, aa3[aa], "H", index, x + offset, 0, 0, icode))
            serial += 1
        if aa != "G":
            lines.append(_atom(serial, "CB", aa3[aa], "H", index, x, 1, 0, icode))
            serial += 1
    for index, aa in enumerate("KF", 1):
        x = float((index - 1) * 3)
        for atom_name, offset in (("N", -0.5), ("CA", 0.0), ("C", 0.5), ("O", 1.0)):
            lines.append(_atom(
                serial, atom_name, aa3[aa], "P", index, x + offset, 0, 3))
            serial += 1
        lines.append(_atom(serial, "CB", aa3[aa], "P", index, x, 1, 3))
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="ascii")


def test_extracts_explicit_h3_heavy_atom_contacts_and_insertion_codes(tmp_path):
    pdb = tmp_path / "toy.pdb"
    _write_toy_complex(pdb)
    interface = extract_h3_interface(
        str(pdb), "H", "P", expected_h3_sequence="DYSA",
        expected_peptide_sequence="KF")
    assert interface["h3_sequence"] == "DYSA"
    assert interface["h3_residues"][1]["icode"] == "A"
    assert interface["contacts"]
    assert all(contact.min_heavy_atom_distance <= 4.5 for contact in interface["contacts"])
    assert all(contact.h3_residue.chain == "H" for contact in interface["contacts"])
    assert all(contact.epitope_residue.chain == "P" for contact in interface["contacts"])
    assert serialize_interface(interface)["contacts"][0]["h3_residue"]["chain"] == "H"


def test_score_has_no_native_identity_input(tmp_path):
    pdb = tmp_path / "toy.pdb"
    _write_toy_complex(pdb)
    interface = extract_h3_interface(str(pdb), "H", "P")
    native = score_h3_sequence("DYSA", interface)
    shuffled = score_h3_sequence("SYDA", interface)
    assert "identity_to_native" not in native
    assert native["n_sidechain_contacts"] > 0
    assert native["chemistry_score"] != shuffled["chemistry_score"]


def test_requires_configured_chain_and_expected_sequences(tmp_path):
    pdb = tmp_path / "toy.pdb"
    _write_toy_complex(pdb)
    with pytest.raises(ValueError, match="Missing configured chains"):
        extract_h3_interface(str(pdb), "X", "P")
    with pytest.raises(ValueError, match="H3 sequence mismatch"):
        extract_h3_interface(str(pdb), "H", "P", expected_h3_sequence="AAAA")


def test_standardized_backbone_has_contiguous_residue_mapping(tmp_path):
    pdb = tmp_path / "toy.pdb"
    standardized = tmp_path / "standardized.pdb"
    _write_toy_complex(pdb)
    manifest = write_standardized_backbone(str(pdb), str(standardized), ["H", "P"])
    heavy = manifest["chains"]["H"]
    assert heavy["length"] == len(heavy["sequence"])
    assert [row["standardized_resseq"] for row in heavy["residue_mapping"]] == list(
        range(1, heavy["length"] + 1))
    insertion = next(
        row for row in heavy["residue_mapping"] if row["original"]["icode"] == "A")
    assert insertion["standardized_resseq"] == 73
    atom_resseqs = {
        int(line[22:26]) for line in standardized.read_text(encoding="ascii").splitlines()
        if line.startswith("ATOM") and line[21] == "H"
    }
    assert atom_resseqs == set(range(1, heavy["length"] + 1))
