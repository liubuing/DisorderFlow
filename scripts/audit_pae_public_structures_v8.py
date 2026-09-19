"""Repeat sequence isolation after coordinate-derived Chothia annotation."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_pae_public_sequences_v8 as seq

PARENT = ROOT / 'data/pae_public_crosscheck_v8'
OUT = PARENT / 'structure_sequence_audit'


def main():
    OUT.mkdir(exist_ok=True)
    manifest = PARENT / 'structures/structural_manifest.json'
    rows = seq.read(manifest)['records']
    seq.freeze(OUT / 'sequence_candidates.json', {'classification': 'coordinate_derived_Chothia_sequences', 'records': rows,
        'structural_manifest_sha256': seq.digest(manifest)})
    seq.freeze(OUT / 'sequence_protocol.json', {'classification': 'retrospective_coordinate_sequence_audit',
        'reference_sha256': seq.digest(ROOT / 'data/pae_independent_v7/reference_union.json'),
        'candidates_sha256': seq.digest(OUT / 'sequence_candidates.json'),
        'script_sha256': seq.digest(Path(seq.__file__)), 'wrapper_sha256': seq.digest(Path(__file__)),
        'thresholds': seq.AXES, 'coverage': .8, 'labels_accessed': False})
    seq.OUT = OUT
    if not (OUT / 'sequence_audit.json').exists():
        seq.run()
    audit = seq.read(OUT / 'sequence_audit.json')
    self_hits = {}
    with tempfile.TemporaryDirectory(prefix='pae_v8_self_') as temporary:
        temp = Path(temporary)
        for axis, (field, threshold) in seq.AXES.items():
            q = temp / (axis+'.fasta')
            seq.write_fasta(q, rows, field, 'instance')
            self_hits[axis], _ = seq.run_mmseqs_search('mmseqs', q, q, temp / (axis+'.tsv'), temp / ('work_'+axis), threshold, 4)
    mm = [r['instance'] for r in audit['rows'] if r['passes_mmseqs']]
    both = [r['instance'] for r in audit['rows'] if r['passes_both']]
    # Add short-sequence sensitivity edges, separately from the original MMseqs graph.
    sensitivity = {k:list(v) for k,v in self_hits.items()}
    for axis, field, threshold in [('h3','cdr_h3_sequence',.5), ('antigen','antigen_sequence',.3)]:
        for a in rows:
            for b in rows:
                if seq.ungapped(a[field], b[field], threshold):
                    sensitivity[axis].append({'query':a['instance'], 'target':b['instance'], 'identity':threshold})
    # Cluster all retained structural records first so excluded bridging members
    # cannot falsely split one component into independent units.
    all_ids = [r['instance'] for r in rows]
    mm_components = seq.connected_components(all_ids, self_hits)
    combined_components = seq.connected_components(all_ids, sensitivity)
    result = {'classification':'retrospective_public_structural_feasibility_not_validation',
        'structural_records':len(rows), 'mmseqs_passing_instances':mm, 'combined_passing_instances':both,
        'mmseqs_all_components':mm_components, 'combined_all_components':combined_components,
        'mmseqs_clean_components':[c for c in mm_components if set(c) <= set(mm)],
        'combined_clean_components':[c for c in combined_components if set(c) <= set(both)],
        'minimum_components':6, 'pretraining_audit_complete':False,
        'new_teacher_scoring_started':False,
        'boundary':'Prior reference is conservative mixed training/structure union. Short matches are not proof of evolutionary homology. Neither component count alone nor exact-ID novelty establishes validation.'}
    seq.freeze(OUT / 'components.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
