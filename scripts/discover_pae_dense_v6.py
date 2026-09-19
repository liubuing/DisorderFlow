"""Metadata-only refresh of the existing temporal PAE discovery window."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_aayl_original_endpoint_cohort import freeze
from scripts.benchmark_aayl_learnability import digest
from scripts.build.acquire_rcsb_candidate_interface_extension import acquire
from scripts.build.discover_rcsb_candidate_interface_extension import discover

OUT=ROOT/'data/pae_external_refresh_v6'
old=ROOT/'data/pae_surrogate_external_v4'
query=json.loads((old/'query.json').read_text())
for node in query['query']['nodes']:
    if node.get('parameters',{}).get('attribute')=='rcsb_accession_info.initial_release_date':
        node['parameters']['value'].update({'from':'2026-09-02','to':'2026-09-17'})
freeze(OUT/'query.json',query)
freeze(OUT/'protocol.json',{'classification':'metadata_refresh_only_not_independent_validation',
    'discovery_contract':{'query_config_sha256':digest(OUT/'query.json')},
    'comparison_snapshot_sha256':digest(old/'snapshot/entries.json'),
    'rule':'Only new metadata IDs can proceed to structural eligibility and unchanged five-axis isolation; no new labels'})
if not (OUT/'snapshot/acquisition.json').exists(): acquire(OUT/'query.json',OUT/'protocol.json',OUT/'snapshot')
if not (OUT/'discovery.json').exists(): discover(OUT/'snapshot',OUT/'discovery.json',include_viral=True)
past={r['rcsb_id'] for r in json.loads((old/'snapshot/entries.json').read_text())['entries']}
present={r['rcsb_id'] for r in json.loads((OUT/'snapshot/entries.json').read_text())['entries']}
freeze(OUT/'delta.json',{'previous_count':len(past),'current_count':len(present),'new_ids':sorted(present-past),
    'new_independent_components_established':0,'status':'no_new_metadata_ids' if not present-past else 'new_metadata_requires_structure_and_isolation_audit'})
print((OUT/'delta.json').read_text())
