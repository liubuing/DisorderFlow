"""Acquire a new, model-free temporal external pool for revision v4.

No model selection, generation, or teacher labeling occurs in this script.
The model development cohort cannot be relabeled as blind confirmation.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.harden_pae_surrogate_v4 import read, write, digest
from scripts.build.acquire_rcsb_candidate_interface_extension import acquire
from scripts.build.discover_rcsb_candidate_interface_extension import discover

OUT = ROOT/'data/pae_surrogate_external_v4'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    query = read(ROOT/'configs/candidate_interface_extension_rcsb_query_v2.json')
    for node in query['query']['nodes']:
        if node.get('parameters',{}).get('attribute')=='rcsb_accession_info.initial_release_date':
            node['parameters']['value'].update({'from':'2026-09-02','to':'2026-09-16'})
    query_path = OUT/'query.json'
    if not query_path.exists():
        write(query_path,query)
    protocol = {'classification':'model_free_temporal_feasibility_not_validation',
        'discovery_contract':{'query_config_sha256':digest(query_path)},
        'minimum_components':6,'minimum_antigen_families':3,
        'isolation':{'coverage':0.8,'vh':0.9,'vl':0.9,'paired_cdr':0.7,'h3':0.5,'antigen':0.3},
        'selection':'All structurally eligible, five-axis isolated components; never select by model/teacher scores.',
        'access_policy':'Freeze models and analysis before any new labels; insufficient cohort yields feasibility failure, not relaxed thresholds.'}
    protocol_path = OUT/'protocol.json'
    if not protocol_path.exists():
        write(protocol_path,protocol)
    if not (OUT/'snapshot/acquisition.json').exists():
        acquire(query_path,protocol_path,OUT/'snapshot')
    if not (OUT/'discovery.json').exists():
        discover(OUT/'snapshot',OUT/'discovery.json',include_viral=True)
    discovery = read(OUT/'discovery.json')
    # Update exclusions with every previously examined structural record.
    base_path = ROOT/'data/successor_v3_confirmatory/reference_union_manifest_v3.json'
    ref = read(base_path)
    added = read(ROOT/'data/candidate_interface_external_calibration_v1/merged_structural_manifest.json')['records']
    existing = {r['reference_id'] for r in ref['records']}
    for r in added:
        rid = 'pae_v4_prior_'+r['instance']
        if rid not in existing:
            ref['records'].append({**r,'reference_id':rid})
    ref['exact_exposed_pdb_ids'] = sorted(set(ref['exact_exposed_pdb_ids'])|{r['pdb_id'].lower() for r in added})
    ref['revision_v4_parent_sha256'] = digest(base_path)
    ref['classification'] = 'exclusion_reference_including_all_previous_PAE_structures'
    if not (OUT/'reference_union.json').exists():
        write(OUT/'reference_union.json',ref)
    print(json.dumps(discovery['counts'],indent=2))


if __name__=='__main__':
    main()
