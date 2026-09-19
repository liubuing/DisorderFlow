"""Recover exact-sequence source records and prepare a provisional factorial pilot.

No assay submission, order, measurement, imputation or cross-batch label merging.
"""
import argparse
import json
import sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_aayl_learnability import load,digest,COHORT
from scripts.build_aayl_original_endpoint_cohort import freeze
from scripts.crosswalk_abbibench_aayl_endpoints import original_rows,finite_value

OUT=ROOT/'results/aayl_remeasurement_v1'
PANEL=ROOT/'results/aayl_failure_v3/measurement_repair_panel.json'
SOURCE=ROOT/'data/abbibench_sequence_mapping_v1/original_alpha_seq'


def prepare():
    schema=json.loads((SOURCE/'schema_audit.json').read_text())
    freeze(OUT/'protocol.json',{'status':'source_recovery_and_provisional_assay_preparation_only',
       'cohort_sha256':digest(COHORT),'panel_sha256':digest(PANEL),'script_sha256':digest(Path(__file__)),
       'sources':[{k:r[k] for k in ['file','sha256','revision']} for r in schema['rows']],
       'matching':'exact heavy AND light sequence, MIT_Target; retain POI, assay and replicate identities',
       'same_assay_complete':'dataset1, family49 assay1/family51 assay2, exactly replicate IDs 1/2/3 finite, one matching POI group only',
       'other_sources':'dataset2 or other assay retained separately; never substitute/merge into original endpoints',
       'pilot':'provisional 3 triples per lineage, all 8 factorial constructs including parent; choose smallest marginal sequence count then most incomplete anchors then POI; no affinity-based selection',
       'replicates':'three independent observations proposed, subject to actual platform; not a lab-specific plate map',
       'limitations':'pilot and old triples are development, not independent efficacy validation; no physical measurements executed'})


def source_recovery(protocol,panel):
    wanted={(p['heavy_sequence'],p['light_sequence']):i for i,p in enumerate(panel)}
    assert len(wanted)==len(panel)
    groups=defaultdict(list)
    for source in protocol['sources']:
        path=SOURCE/source['file']
        assert digest(path)==source['sha256']
        count=0
        for row in original_rows(path):
            if row['Target']!='MIT_Target': continue
            index=wanted.get((row['HC'],row['LC']))
            if index is None: continue
            groups[(index,source['file'],row['Assay'],row['POI'])].append({
                'replicate':row['Replicate'],'original_log10_nM':finite_value(row['Pred_affinity'])})
            count+=1
        print(f"Scanned {source['file']}: {count} exact-sequence target records",flush=True)
    records=[]
    for i,p in enumerate(panel):
        matched=[]
        expected='1' if p['family']=='AAYL49' else '2'
        for (idx,source,assay,poi),obs in groups.items():
            if idx!=i: continue
            same=source==protocol['sources'][0]['file'] and assay==expected
            ids=[o['replicate'] for o in obs]
            complete=same and len(ids)==3 and set(ids)=={'1','2','3'} and all(o['original_log10_nM'] is not None for o in obs)
            matched.append({'source':source,'assay':assay,'poi':poi,'same_original_assay':same,
                            'complete_original_three_replicates':complete,'observations':obs})
        relevant=[g for g in matched if g['same_original_assay']]
        recoverable=len(relevant)==1 and relevant[0]['complete_original_three_replicates']
        records.append({'panel_index':i,'family':p['family'],'role':p['role'],
            'recoverable_same_assay':recoverable,'same_assay_group_count':len(relevant),
            'other_assay_group_count':sum(not g['same_original_assay'] for g in matched),'matches':matched})
    result={'status':'historical_source_audit_not_new_measurements','records':records,
       'summary':{'panel_candidates':len(panel),'recoverable_same_assay':sum(r['recoverable_same_assay'] for r in records),
       'any_same_assay_match':sum(r['same_assay_group_count']>0 for r in records),
       'any_other_assay_match':sum(r['other_assay_group_count']>0 for r in records),
       'no_exact_match_either_source':sum(not r['matches'] for r in records),
       'original_118_incomplete_now_recoverable':sum(r['recoverable_same_assay'] and panel[r['panel_index']]['status']=='incomplete_source_readings' for r in records)}}
    freeze(OUT/'source_recovery.json',result)
    return result


def factorial(parent,changes):
    result=set()
    for n in range(4):
        for subset in combinations(sorted(changes),n):
            seq=list(parent)
            for pos,aa in subset: seq[pos]=aa
            result.add(''.join(seq))
    assert len(result)==8
    return result


def pilot(d,rows,changes,panel):
    constructs=[]; selected=[]
    for family,s in sorted(d['structures'].items()):
        parent=s['sequences'][s['heavy_chain']]; light=s['sequences'][s['light_chain']]
        triples=[i for i,r in enumerate(rows) if r['family']==family and len(changes[i])==3]
        cubes={i:factorial(parent,changes[i]) for i in triples}
        incomplete={p['heavy_sequence'] for p in panel if p['family']==family and p['status']=='incomplete_source_readings'}
        chosen=[]; used=set()
        for _ in range(3):
            idx=min((i for i in triples if i not in chosen),key=lambda i:(len(cubes[i]-used),-len(cubes[i]&incomplete),rows[i]['poi']))
            chosen.append(idx); used|=cubes[idx]
        sequence_ids={seq:f"{family}_P{n:03d}" for n,seq in enumerate(sorted(used,key=lambda seq:(sum(a!=b for a,b in zip(seq,parent)),seq)),1)}
        for seq,identifier in sequence_ids.items():
            muts=[{'observed_heavy_position_1based':j+1,'parent':b,'mutant':a} for j,(a,b) in enumerate(zip(seq,parent)) if a!=b]
            constructs.append({'construct_id':identifier,'family':family,'heavy_sequence':seq,'light_sequence':light,
                'h3_sequence':''.join(seq[j] for j in s['h3_indices_zero_based_observed_heavy']),
                'mutation_count':len(muts),'mutations':muts,'role':'parent_control' if not muts else 'factorial_variant',
                'target_sequence':next(iter(s['antigen_chain_sequences'].values())),
                'historical_complete_reading':any(r['family']==family and r['heavy_sequence']==seq for r in rows),
                'associated_triples':[rows[i]['poi'] for i in chosen if seq in cubes[i]],'measurement_status':'not_measured_in_this_pilot'})
        for i in chosen:
            selected.append({'family':family,'triple_poi':rows[i]['poi'],'factorial_construct_ids':sorted(sequence_ids[seq] for seq in cubes[i])})
    result={'status':'provisional_pending_platform_and_budget_not_submitted','constructs':constructs,'selected_triples':selected,
        'unique_constructs':len(constructs),'proposed_independent_replicates':3,'proposed_variant_observations':3*len(constructs),
        'excluded_from_count':'platform-specific blanks, specificity/reference controls, titration concentrations, failed runs and QC repeats',
        'assay_notes':'Remeasure complete 8-member sets in matched conditions; old complete values do not replace same-batch controls; no biological acceptance threshold assumed'}
    freeze(OUT/'pilot_design.json',result)
    lines=[]
    for c in constructs:
        for chain in ['heavy','light']:
            lines.extend([f">{c['construct_id']}|{chain}|family={c['family']}",c[f'{chain}_sequence']])
    fasta='\n'.join(lines)+'\n'; path=OUT/'pilot_variable_domains.fasta'
    if path.exists() and path.read_text()!=fasta: raise ValueError('Refusing FASTA overwrite')
    path.write_text(fasta)
    freeze(OUT/'blank_readout_template.json',{'status':'empty_template_not_results',
       'fields':'estimate and units must be supplied from actual assay; censored/missing cannot be assigned fabricated values',
       'records':[{'construct_id':c['construct_id'],'replicate':rep,'assay':None,'batch':None,'target':None,
           'estimate':None,'units':None,'readout_status':'pending','censoring_limit':None,'raw_data_file':None}
           for c in constructs for rep in [1,2,3]]})
    return result


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert digest(COHORT)==protocol['cohort_sha256'] and digest(PANEL)==protocol['panel_sha256']
    assert digest(Path(__file__))==protocol['script_sha256']
    panel=json.loads(PANEL.read_text())['records']
    recovery=source_recovery(protocol,panel)
    d,rows,changes=load()
    design=pilot(d,rows,changes,panel)
    print(json.dumps({'recovery':recovery['summary'],'pilot_constructs':design['unique_constructs'],
       'proposed_observations':design['proposed_variant_observations']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.run: run()
