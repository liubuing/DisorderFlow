"""Supplement SAbDab refresh with direct RCSB antibody text discovery."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
from scripts.audit_pae_public_sources_v8 import normalize
from scripts.build.acquire_rcsb_candidate_interface_extension import post_json,result_ids,fetch_entries,SEARCH_URL
OUT=ROOT/'data/pae_family_refresh_v21/rcsb_direct'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    short=read(OUT.parent/'rcsb_query.json')['query']
    terms=['antibody','Fab','immunoglobulin','nanobody','VHH']
    query={'query':{'type':'group','logical_operator':'and','nodes':[short,{'type':'group','logical_operator':'or','nodes':[{'type':'terminal','service':'full_text','parameters':{'value':t}} for t in terms]}]},'return_type':'entry','request_options':{'return_all_hits':True,'results_content_type':['experimental']}}
    freeze(OUT/'protocol.json',{'classification':'supplementary_metadata_search_no_scoring','query':query,
        'scope':'allrelease dates; peptide5-50AA unchanged; inspect hits absent SAbDab pairing or absent v8reference; keywords not proof of antibody or antigen relationship',
        'script_sha256':digest(Path(__file__))})
    if not (OUT/'search.json').exists():freeze(OUT/'search.json',post_json(SEARCH_URL,query))
    ids={normalize(p) for p in result_ids(read(OUT/'search.json'))}
    sab=list(csv.DictReader((OUT.parent/'sabdab_all_summary.csv').open(encoding='utf-8-sig')))
    paired={normalize(r['PDB']) for r in sab if r['Hchain'] not in ('','NA') and r['Lchain'] not in ('','NA')}
    refs={normalize(r['pdb_id']) for r in read(ROOT/'data/pae_independent_v7/reference_union.json')['records']}
    prior={r['pdb_id'] for r in read(ROOT/'data/pae_public_crosscheck_v8/audit.json')['rows']}
    selected=sorted(ids-paired-refs-prior)
    print('Direct RCSB text+short hits',len(ids),'not already covered',len(selected),flush=True)
    entries=[]
    for offset in range(0,len(selected),100):
        path=OUT/f'entries_{offset:05d}.json'
        if not path.exists():freeze(path,{'entries':fetch_entries([p.upper() for p in selected[offset:offset+100]])})
        entries.extend(read(path)['entries'])
    rows=[]
    for e in entries:
        polymers=[]
        for ent in e.get('polymer_entities') or []:
            poly=ent.get('entity_poly') or {};seq=''.join(poly.get('pdbx_seq_one_letter_code_can','').split())
            if poly.get('rcsb_entity_polymer_type')!='Protein':continue
            polymers.append({'id':ent['rcsb_id'],'description':(ent.get('rcsb_polymer_entity') or {}).get('pdbx_description'),
                'length':len(seq),'sequence':seq,'chains':(ent.get('rcsb_polymer_entity_container_identifiers') or {}).get('auth_asym_ids')})
        res=(e.get('rcsb_entry_info') or {}).get('resolution_combined') or []
        methods=[x['method'] for x in e.get('exptl') or []]
        shortpoly=[p for p in polymers if 5<=p['length']<=50]
        longpoly=[p for p in polymers if p['length']>=90]
        plausible=bool(shortpoly and len(longpoly)>=2 and res and min(res)<=4 and any(m in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY'] for m in methods))
        rows.append({'pdb_id':normalize(e['rcsb_id']),'title':e['struct']['title'],'release_date':e['rcsb_accession_info']['initial_release_date'],
            'polymers':polymers,'resolution':res,'methods':methods,'needs_antibody_chain_review':plausible,'independence_established':False})
    freeze(OUT/'audit.json',{'text_short_hits':len(ids),'previously_uncovered':len(selected),'rows':rows,
        'chain_review_candidates':[r['pdb_id'] for r in rows if r['needs_antibody_chain_review']],
        'boundary':'chain count heuristics only; short chain may be tag, ligand or other protein fragment; no new independent family established'})
    print(json.dumps(read(OUT/'audit.json'),indent=2),flush=True)

if __name__=='__main__':main()
