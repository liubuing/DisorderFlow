"""Separate Ridge training, generator data, short matches and chemical validity."""
import json
import pickle
import random
import sys
from pathlib import Path

import lmdb
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_pae_dense_v6 import read, freeze, digest
from scripts.audit_pae_public_sources_v8 import normalize

OUT = ROOT / 'data/pae_public_provenance_v9'
SOURCE = ROOT / 'data/pae_public_crosscheck_v8'
STANDARD = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}


def best_overlap(a, b):
    """All ungapped offsets with >=80% coverage on both sequences."""
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < .8:
        return None
    best = None
    for offset in range(-len(b)+1, len(a)):
        start, stop = max(0, offset), min(len(a), offset+len(b))
        n = stop-start
        if n / len(a) < .8 or n / len(b) < .8:
            continue
        matches = sum(a[i] == b[i-offset] for i in range(start, stop))
        hit = {'identity':matches/n, 'matches':matches, 'aligned_length':n,
            'query_coverage':n/len(a), 'reference_coverage':n/len(b), 'offset':offset}
        if best is None or hit['identity'] > best['identity']:
            best = hit
    return best


def chemistry(chain):
    residues = [r for r in chain if 'CA' in r]
    modified = [{'residue':r.resname, 'number':r.id[1], 'hetero_flag':r.id[0]} for r in residues if r.resname not in STANDARD]
    return {'resolved_ca_residues':len(residues), 'nonstandard_residues':modified,
        'canonical_sequence_with_unknown_markers': ''.join(seq1(r.resname, undef_code='X') for r in residues),
        'supported_unmodified_peptide':not modified}


def load_lmdb(path):
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    rows = []
    with env.begin() as txn:
        for key, value in txn.cursor():
            r = pickle.loads(value)
            rows.append({k:r.get(k,'') for k in ('id','pdb_id','vh_sequence','vl_sequence','antigen_sequence','cdr_h3_sequence')})
    env.close()
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    groups = read(SOURCE / 'retrospective_candidates.json')['components']
    split = read(ROOT / 'data/candidate_interface_multiscaffold_v1/split_manifest.json')
    hold = read(ROOT / 'data/multiscaffold_confirmatory_v2/holdout_manifest.json')
    train_dev = []
    for c in hold['components']:
        name = next(k for k in ('train','calibration','test') if c['component_id'] in split[k])
        # Only sequence metadata from the historical test manifest, never its labels.
        train_dev.append({**c['representative'], 'component_id':c['component_id'], 'usage':'ridge_'+name})
    for c in read(ROOT / 'data/candidate_interface_external_calibration_v1/extension_manifest_v1.json')['components']:
        train_dev.append({**c, 'usage':'exposed_transfer_development'})
    train_dev = [r for r in train_dev if r['usage'] != 'ridge_test']
    paths = ['data/phase3_v5_1_pair_clustered/train.lmdb', 'data/phase3_v5_1_pair_clustered/val.lmdb']
    freeze(OUT / 'protocol.json', {'classification':'retrospective_data_and_chemistry_audit_no_labels',
        'candidate_sha256':digest(SOURCE / 'retrospective_candidates.json'),
        'ridge_split_sha256':digest(ROOT / 'data/candidate_interface_multiscaffold_v1/split_manifest.json'),
        'lineage_sha256':digest(ROOT / 'publication/multiscaffold_v2_calibration_ext_checkpoint_lineage.json'),
        'generator_dataset_hashes':{p:digest(ROOT / p) for p in paths},
        'null_permutations':1000, 'seed':20260918,
        'null_scope':'composition-preserving query shuffle, max ungapped identity across original15 Ridge training antigens only; descriptive, not homology test or new acceptance gate',
        'chemistry':'any resolved CA residue outside20standard residues blocks unchanged20AA peptide scoring; no silent deletion or dephosphorylation',
        'script_sha256':digest(Path(__file__))})
    generator_sets = {p:load_lmdb(ROOT / p) for p in paths}
    freeze(OUT / 'generator_sequence_inventory.json', {'records':generator_sets,
        'scope':'configured stageA train/val datasets; not proof of per-sample checkpoint exposure; stageB disorder data not exhaustively audited'})
    parser = MMCIFParser(QUIET=True)
    rng = random.Random(20260918)
    rows = []
    for g in groups:
        r = g['representative']
        query = r['antigen_sequence']
        model = next(parser.get_structure(r['pdb_id'], str(ROOT / r['source_cif'])).get_models())
        chem = chemistry(model[r['original_chain_ids']['antigen']])
        matches = []
        for ref in train_dev:
            target = ref['antigen_sequence']
            hit = best_overlap(query, target)
            matches.append({'component_id':ref['component_id'], 'pdb_id':ref.get('pdb_id'), 'usage':ref['usage'],
                'antigen_sequence':target, 'exact':query==target, 'query_contained_in_reference':query in target,
                'reference_contained_in_query':target in query, 'ungapped_both80':hit})
        generators = {}
        member_pdbs = {normalize(m.split('_')[0]) for m in g['members']}
        for path, refs in generator_sets.items():
            generators[path] = {
                'exact_pdb_hits':[ref['id'] for ref in refs if normalize(ref['pdb_id']) in member_pdbs],
                'exact_vh_hits':[ref['id'] for ref in refs if ref['vh_sequence']==r['vh_sequence']],
                'exact_vl_hits':[ref['id'] for ref in refs if ref['vl_sequence']==r['vl_sequence']],
                'antigen_exact_hits':[ref['id'] for ref in refs if ref['antigen_sequence']==query],
                'antigen_full_query_substring_hits':[{'id':ref['id'],'reference_length':len(ref['antigen_sequence'])} for ref in refs if query in ref['antigen_sequence']]}
        train_ag = [ref['antigen_sequence'] for ref in train_dev if ref['usage']=='ridge_train']
        def maximum(seq):
            return max(((best_overlap(seq,t) or {}).get('identity',0) for t in train_ag), default=0)
        observed = maximum(query)
        shuffled = []
        for _ in range(1000):
            chars = list(query)
            rng.shuffle(chars)
            shuffled.append(maximum(''.join(chars)))
        rows.append({'component_id':g['component_id'], 'pdb_id':r['pdb_id'], 'antigen_sequence_used_in_v8':query,
            'source_cif_sha256_verified':digest(ROOT/r['source_cif'])==r['source_cif_sha256'],
            'chemistry':chem, 'ridge_train_development_comparisons':matches, 'generator_dataset_hits':generators,
            'descriptive_shuffle':{'max_train_identity':observed,'fraction_null_ge_observed_plus1':(1+sum(v>=observed for v in shuffled))/1001,
                'fraction_null_max_ge_0_3':sum(v>=.3 for v in shuffled)/1000, 'not_a_homology_pvalue':True}})
    freeze(OUT / 'audit.json', {'classification':'provenance_and_chemistry_not_performance', 'rows':rows,
        'no_new_PAE_labels':True, 'generator_upstream_exhaustively_audited':False,
        'af2_multimer_v3_cutoff':'2021-09-30',
        'af2_source':'https://github.com/google-deepmind/alphafold/blob/main/docs/technical_note_v2.3.0.md',
        'af2_exact_training_membership':'unknown; all seven predate cutoff, temporal exclusion not established',
        'proteinmpnn_exact_training_membership':'unknown; no full matching upstream training list audited'})
    for r in rows:
        print(json.dumps({'component':r['component_id'],'pdb':r['pdb_id'], 'chemistry':r['chemistry'],
            'ridge_exact_or_substring':[{k:m[k] for k in ('component_id','usage','antigen_sequence')} for m in r['ridge_train_development_comparisons'] if m['query_contained_in_reference'] or m['reference_contained_in_query']],
            'generator_hit_counts':{p:{k:len(v) for k,v in h.items()} for p,h in r['generator_dataset_hits'].items()},
            'shuffle':r['descriptive_shuffle']}), flush=True)


if __name__ == '__main__':
    main()
