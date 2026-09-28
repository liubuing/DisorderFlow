"""Refresh public metadata; expand annotation coverage, not peptide length scope."""
import csv
import io
import json
import sys
import urllib.request
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
from scripts.audit_pae_public_sources_v8 import normalize
from scripts.build.acquire_rcsb_candidate_interface_extension import post_json,result_ids,fetch_entries,SEARCH_URL
OUT=ROOT/'data/pae_family_refresh_v21'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze(OUT/'protocol.json',{'classification':'new_family_feasibility_before_any_new_labels',
        'scope':'pairedH/L,actual antigen5-50AA,experimentalresolution<=4A; all SAbDab antigen annotations checked, not only PEPTIDE',
        'isolation':'retain prior five-axis thresholds VH/VL0.9 pairedCDR0.7 H3 0.5 antigen0.3 bothcoverage0.8 plus conservative ungapped sensitivity and fragment containment; biological family review required',
        'reference':'all earlier training/development/exposure plus publicv11; no relabeling of exposed pool as independent',
        'gate':'coordinates/chemistry/H3contact, actual training overlap and upstream provenance reviewed before new generation or AF2; no score-based selection',
        'no_training_or_AF2_scoring':True,'script_sha256':digest(Path(__file__))})
    snapshot=OUT/'sabdab_all_summary.csv'
    if not snapshot.exists():
        url='https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary'
        request=urllib.request.Request(url,headers={'User-Agent':'DisorderFlow-public-family-audit'})
        with urllib.request.urlopen(request,timeout=120) as response:data=response.read()
        if not data.startswith(b'INSTANCE,'):raise ValueError('Unexpected SAbDab CSV response')
        snapshot.write_bytes(data)
        freeze(OUT/'download_receipt.json',{'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':digest(snapshot)})
    sab=list(csv.DictReader(io.StringIO(snapshot.read_text(encoding='utf-8-sig'))))
    bypdb={}
    for r in sab:
        if r['Hchain'] not in ('','NA') and r['Lchain'] not in ('','NA'):
            bypdb.setdefault(normalize(r['PDB']),[]).append(r)
    query={'query':{'type':'terminal','service':'text','parameters':{'attribute':'entity_poly.rcsb_sample_sequence_length','operator':'range','value':{'from':5,'to':50,'include_lower':True,'include_upper':True}}},
        'return_type':'entry','request_options':{'return_all_hits':True,'results_content_type':['experimental']}}
    freeze(OUT/'rcsb_query.json',query)
    if not (OUT/'rcsb_short_search.json').exists():freeze(OUT/'rcsb_short_search.json',post_json(SEARCH_URL,query))
    ids={x.lower() for x in result_ids(read(OUT/'rcsb_short_search.json'))}
    selected=sorted(ids&set(bypdb))
    print('SAbDab rows',len(sab),'pairedHL PDBs',len(bypdb),'short-chain intersect',len(selected),flush=True)
    entries=[]
    for offset in range(0,len(selected),100):
        path=OUT/f'entries_{offset:05d}.json'
        if not path.exists():freeze(path,{'entries':fetch_entries([p.upper() for p in selected[offset:offset+100]])})
        entries.extend(read(path)['entries'])
        print('metadata',min(offset+100,len(selected)),'/',len(selected),flush=True)
    prior=read(ROOT/'data/pae_public_crosscheck_v8/audit.json')
    priorids={r['pdb_id'] for r in prior['rows']}
    ref=read(ROOT/'data/pae_independent_v7/reference_union.json')
    refids={normalize(r['pdb_id']) for r in ref['records']}
    candidates=[];eligible=[]
    for e in entries:
        pdb=normalize(e['rcsb_id']); resolution=(e.get('rcsb_entry_info') or {}).get('resolution_combined') or []
        methods=[r['method'] for r in e.get('exptl') or []]
        if not resolution or min(resolution)>4 or not any(m in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY'] for m in methods):continue
        for r in bypdb[pdb]:
            for chain,kind in zip(r['antigen_chain'].split('|'),r['antigen_type'].split('|')):
                for ent in e.get('polymer_entities') or []:
                    ident=ent.get('rcsb_polymer_entity_container_identifiers') or {}
                    ep=ent.get('entity_poly') or {}
                    if chain not in (ident.get('auth_asym_ids') or []) or ep.get('rcsb_entity_polymer_type')!='Protein':continue
                    seq=''.join(ep.get('pdbx_seq_one_letter_code_can','').split())
                    if not 5<=len(seq)<=50:continue
                    item={'instance':r['INSTANCE']+'-'+chain,'pdb_id':pdb,'chains':{'heavy':r['Hchain'],'light':r['Lchain'],'antigen':chain},
                        'antigen_annotation':kind,'antigen_name':r['antigen_name'],'antigen_description':(ent.get('rcsb_polymer_entity') or {}).get('pdbx_description'),
                        'title':e['struct']['title'],'release_date':e['rcsb_accession_info']['initial_release_date'],'resolution':min(resolution),
                        'vh_sequence':r['VH'],'vl_sequence':r['VL'],'cdr_h3_sequence':r['CDR-H3'],
                        'paired_cdr_sequence':''.join(r[k] for k in ['CDR-H1','CDR-H2','CDR-H3','CDR-L1','CDR-L2','CDR-L3']),
                        'antigen_sequence':seq,'in_prior_v8':pdb in priorids,'in_prior_reference':pdb in refids,
                        'independence_established':False}
                    eligible.append(item)
                    if pdb not in priorids and pdb not in refids:candidates.append(item)
    unique=lambda rows:list({r['instance']:r for r in rows}.values())
    eligible=unique(eligible);candidates=unique(candidates)
    freeze(OUT/'discovery.json',{'counts':{'sabdab_rows':len(sab),'paired_pdbs':len(bypdb),'short_chain_intersection':len(selected),
        'eligible_instances':len(eligible),'eligible_pdbs':len({r['pdb_id'] for r in eligible}),
        'new_candidate_instances':len(candidates),'new_candidate_pdbs':len({r['pdb_id'] for r in candidates})},
        'new_candidates':candidates,'eligible_records':eligible,'boundary':'metadata candidates, not independent family validation',
        'snapshot_sha256':digest(snapshot),'teacher_labels_accessed':False})
    print(json.dumps({'counts':read(OUT/'discovery.json')['counts'],'new_candidates':candidates},indent=2),flush=True)

if __name__=='__main__':main()
