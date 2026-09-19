"""Evaluate frozen PAE baselines only after the complete dense teacher run."""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_pae_dense_v6 import OUT,read,freeze,digest
from scripts import pae_sequence_order_v5 as v5
from scripts.build_candidate_interface_multiscaffold_v1 import hierarchical_sem


def main():
    source=read(OUT/'teacher_labels.json')
    if source['incomplete_entities']: raise ValueError('Do not silently shrink teacher panel')
    protocol=read(OUT/'protocol.json')
    for name,sha in protocol['models_sha256'].items(): assert digest(v5.OUT/name)==sha
    data=torch.load(v5.OLD/'features.pt',map_location='cpu',weights_only=False)
    rows=[]; simple=[]
    for r in source['rows']:
        observations=[{'protocol_id':str(o['model']),'af2_seed':o['seed'],'value':o['target']} for o in r['observations']]
        row={'id':r['entity_id'],'scaffold':r['component_id'],'arm':r['arm'],'entity_type':r['entity_type'],
            'h3_sequence':r['h3_sequence'],'target':r['target'],'sem':hierarchical_sem(observations,'value')}
        rows.append(row)
        old_index=next(i for i,old in enumerate(data['rows']) if old['scaffold']==row['scaffold'])
        x=data['simple'][old_index].copy()
        x[:20]=[row['h3_sequence'].count(a)/len(row['h3_sequence']) for a in v5.AA]
        simple.append(x)
    simple=np.asarray(simple)
    xs={'composition_tuned':simple,
        'composition_position_tuned':np.c_[simple,np.array([v5.order_features(r['h3_sequence']) for r in rows])],
        'composition_bigram_tuned':np.c_[simple,np.array([v5.bigrams(r['h3_sequence']) for r in rows])]}
    pred={}
    for name,x in xs.items():
        with np.load(v5.OUT/(name+'.npz')) as z: pred[name]=((x-z['mean'])/z['scale'])@z['coef']+z['intercept']
    mask=np.array([r['split']=='train' for r in data['rows']])
    y=np.array([r['target'] for r in data['rows']],dtype=np.float32)
    model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(data['simple'][mask],y[mask])
    pred['v4_ridge_fixed10']=model.predict(simple)
    models={name:{'generated_only':v5.summarize(v5.by_scaffold(rows,p)),
                 'within_generator':v5.within_arm(rows,p),'all_entities':v5.summarize(v5.by_scaffold(rows,p,False))} for name,p in pred.items()}
    freeze(OUT/'evaluation.json',{'classification':'dense_exposed_development_not_independent_validation',
        'teacher_sha256':digest(OUT/'teacher_labels.json'),'models':models,'new_training_labels_used':False,
        'candidate_count':sum(r['entity_type']=='candidate' for r in rows)})
    freeze(OUT/'predictions.json',{'rows':rows,'models':{k:v.tolist() for k,v in pred.items()}})
    lines=['# PAE 加密候选面板结果','','同一批已暴露六个骨架的开发验证，非新增独立抗原验证。', '',
        '| 模型 | 候选骨架ρ中位数 | top20召回 | 同生成方法组内正确率 |','|---|---:|---:|---:|']
    for name,m in models.items():
        a,b=m['generated_only'],m['within_generator']
        lines.append(f"| {name} | {a['median_rho']:.3f} | {a['macro_top20_recall']:.3f} | {b['macro_reliable_pair_accuracy']:.3f} |")
    lines+=['','所有模型在本轮标签产生前固定；未用新标签选参。失败槽位不会静默剔除。未宣称实验结合性能或端到端加速。','']
    (ROOT/'docs/PAE_DENSE_V6_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()
