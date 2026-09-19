"""Verify AAYL coordinate parents against original published sequences and contacts."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/abbibench_sequence_mapping_v1"
STRUCT = ROOT / "data/abbibench_structural_audit_v1"


def main():
    source = OUT / "engelhart2022.xml"
    xml = ET.parse(source)
    seeds = {}
    for table in xml.findall(".//table-wrap"):
        caption = table.find("caption")
        if caption is None or "Seed" not in "".join(caption.itertext()):
            continue
        for row in table.findall(".//tr"):
            cells = ["".join(c.itertext()).strip() for c in row]
            if len(cells) == 2 and cells[0] in ("Target", "14-VH", "14-VL", "91-VH", "91-VL", "95-VH", "95-VL"):
                seeds[cells[0]] = cells[1]
    if len(seeds) != 7:
        raise ValueError("Published parent table incomplete")
    structures = json.loads((STRUCT / "structural_manifest.json").read_text())["records"]
    matches = []
    for row in structures:
        if "AAYL" not in row["file"]:
            continue
        h, l = row["sequences"][row["heavy_chain"]], row["sequences"][row["light_chain"]]
        hm = [key for key, sequence in seeds.items() if key.endswith("-VH") and sequence == h]
        lm = [key for key, sequence in seeds.items() if key.endswith("-VL") and sequence == l]
        target = list(row["antigen_chain_sequences"].values()) == [seeds["Target"]]
        matches.append({"instance": row["instance"], "datasets": row["datasets"],
                        "heavy_parent_exact_match": hm, "light_parent_exact_match": lm,
                        "antigen_exact_match": target, "coordinate_parent_verified": len(hm) == len(lm) == 1 and target,
                        "structure_type": "AlphaFold3_prediction_as_reported_by_AbBiBench",
                        "structure_source": "https://arxiv.org/pdf/2506.04235v2#page=18",
                        "experimental_sequence_source": "https://www.nature.com/articles/s41597-022-01779-4/tables/2"})
    # Inspect, but do not change, the disputed 5a12 VEGF chain assignment.
    row = next(r for r in structures if r["instance"] == "ABB007")
    model = PDBParser(QUIET=True).get_structure("5a12", str(STRUCT / "structures" / Path(row["file"]).name))[0]
    hr = [r for r in model[row["heavy_chain"]] if r.id[0] == " " and "CA" in r]
    antigen = np.array([a.coord for c in row["antigen_chains"] for a in model[c].get_atoms() if a.element not in ("H", "D")])
    mins = []
    for i in row["h3_indices_zero_based_observed_heavy"]:
        atoms = np.array([a.coord for a in hr[i] if a.element not in ("H", "D")])
        mins.append(float(np.linalg.norm(atoms[:, None] - antigen[None, :], axis=-1).min()))
    payload = {"classification": "source_parent_and_structure_provenance_verification",
               "original_xml_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "original_xml_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9606274/fullTextXML",
               "published_seed_sequences": seeds, "AAYL_matches": matches,
               "5a12_vegf": {"heavy_chain": row["heavy_chain"], "light_chain": row["light_chain"],
                    "antigen_chains": row["antigen_chains"], "minimum_H3_antigen_nonhydrogen_distance_A": min(mins),
                    "H3_positions_with_nonhydrogen_contact_under_5A": sum(x < 5 for x in mins),
                    "auxiliary_metadata_conflict": "epitope/paratope labels conflict; roles verified by antibody numbering and geometry only",
                    "chain_selection_changed": False},
               "label_values_used": False}
    (OUT / "parent_provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"AAYL": matches, "5a12_vegf": payload["5a12_vegf"]}, indent=2))


if __name__ == "__main__":
    main()
