"""Reuse verified canonical design PDBs; avoid recopying full Bio.PDB hierarchies."""
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import generate_multiscaffold_confirmatory_v2 as generation
from scripts.prepare_pae_dense_v6 import OUT,RAW,digest,freeze


def verified_canonical(representative,output_path):
    component=output_path.stem
    old=ROOT/f'data/multiscaffold_confirmatory_v2/generation_work/structures/{component}.pdb'
    source=ROOT/representative['cif_path']
    rows=[r for r in json.loads(RAW.read_text())['attempts'] if r['component_id']==component]
    assert {r['canonical_pdb_sha256'] for r in rows}=={digest(old)}
    assert {r['input_cif_sha256'] for r in rows}=={digest(source)}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(old,output_path)
    return source


if __name__=='__main__':
    freeze(OUT/'generation_execution.json',{'wrapper_sha256':digest(Path(__file__)),
        'generator_sha256':digest(Path(generation.__file__)),'canonical_source':'existing hash-verified design-time PDBs',
        'reason':'old canonicalize deepcopies residue including parent hierarchy; reuse identical canonical bytes'})
    generation.canonicalize_structure=verified_canonical
    generation.main()
