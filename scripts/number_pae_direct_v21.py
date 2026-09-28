"""Sequence-based antibody role review of direct RCSB keyword candidates."""
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import read,freeze,digest
OUT=ROOT/'data/pae_family_refresh_v21/rcsb_direct/numbering'

def main():
    OUT.mkdir(exist_ok=True)
    source=OUT.parent/'audit.json'
    allrows=read(source)['rows']
    keyword=lambda r:bool(re.search(r'\b(antibod\w*|fab|immunoglobulin|nanobod\w*|vhh)\b',r['title']+' '+' '.join(p['description'] or '' for p in r['polymers']),re.I))
    rows=[r for r in allrows if r['needs_antibody_chain_review'] and keyword(r)]
    sequences=sorted({p['sequence'] for r in rows for p in r['polymers'] if 80<=p['length']<=600})
    freeze(OUT/'protocol.json',{'classification':'sequence_role_review_before_structure_and_isolation',
        'source_sha256':digest(source),'selected_entries':len(rows),'unique_sequences':len(sequences),
        'method':'ANARCII antibody accuracy CPU Chothia; sameentry H plusK/L required; biological/structural pairing still unconfirmed',
        'scope':'first pass explicit antibody-related keyword entries; 80-600aa chains; other lengths and nonkeyword hits remain unresolved, not asserted negatives',
        'script_sha256':digest(Path(__file__))})
    from anarcii import Anarcii
    model=Anarcii(seq_type='antibody',mode='accuracy',batch_size=32,cpu=True,ncpu=2,verbose=False)
    labels={}
    for start in range(0,len(sequences),32):
        path=OUT/f'batch_{start:04d}.json'
        if not path.exists():
            batch={'seq_'+hashlib.sha256(s.encode()).hexdigest()[:16]:s for s in sequences[start:start+32]}
            numbered=model.number(batch)
            numbered=model.to_scheme('chothia')
            records={}
            for key,r in numbered.items():
                records[batch[key]]={k:r.get(k) for k in ['chain_type','score','query_start','query_end','error','numbering']}
            freeze(path,{'records':records})
        labels.update(read(path)['records'])
        print('Numbered',len(labels),'/',len(sequences),flush=True)
    output=[]
    for r in rows:
        chains=[]
        for p in r['polymers']:
            if p['sequence'] not in labels:continue
            result=labels[p['sequence']]
            chains.append({'entity':p['id'],'auth_chains':p['chains'],'description':p['description'],'length':p['length'],**result})
        heavy=[c for c in chains if c['chain_type']=='H' and not c['error']]
        light=[c for c in chains if c['chain_type'] in ['K','L'] and not c['error']]
        output.append({'pdb_id':r['pdb_id'],'title':r['title'],'heavy_entities':[c['entity'] for c in heavy],
            'light_entities':[c['entity'] for c in light],'has_paired_roles':bool(heavy and light),'chains':chains})
    freeze(OUT/'audit.json',{'rows':output,'sameentry_paired_candidates':[r['pdb_id'] for r in output if r['has_paired_roles']],
        'deferred_nonkeyword_entries':sum(r['needs_antibody_chain_review'] and not keyword(r) for r in allrows),
        'boundary':'roles not proof of direct peptide binding or family independence'})
    print(json.dumps({'reviewed':len(rows),'paired':[r['pdb_id'] for r in output if r['has_paired_roles']]}),flush=True)

if __name__=='__main__':main()
