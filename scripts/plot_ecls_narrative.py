"""Make evidence-led ECLS figures from saved scores and one fixed input structure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK = '#183448'
TEAL = '#137e78'
RUST = '#be644b'
GOLD = '#d3942d'
GRAY = '#b8c5cb'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect(root=ROOT):
    root = Path(root)
    sources = {
        'temporal': 'results/publication/h3_ecls_temporal_final_v1/results.json',
        'ranking': 'results/publication/h3_candidate_reranking_dev_v1/results.json',
        'calibration': 'results/publication/h3_generator_calibration_v1/analysis.json',
    }
    data = {k: read(root/v) for k,v in sources.items()}
    temporal = data['temporal']
    # Illustration chosen by identifier before considering any effect magnitude.
    record = min(temporal['results'], key=lambda r:r['id'])
    relative = ('results/publication/h3_ecls_temporal_final_v1/work/' + record['id']
                + '/complex/' + record['id'] + '_complex.pdb')
    pdb = root/relative
    coords = {'H': [], 'L': [], 'P': []}
    seen = set()
    for line in pdb.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
            continue
        chain = line[21]
        key = (chain, line[22:27])
        if chain not in coords or key in seen:
            continue
        seen.add(key)
        coords[chain].append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    all_xyz = np.concatenate([np.asarray(v) for v in coords.values() if v])
    center = all_xyz.mean(axis=0)
    basis = np.linalg.svd(all_xyz-center, full_matrices=False)[2][:2].T
    projected = {k:((np.asarray(v)-center)@basis).tolist() for k,v in coords.items() if v}
    result = {
        'source_sha256': {v:hashlib.sha256((root/v).read_bytes()).hexdigest() for v in sources.values()},
        'illustration': {'record_id':record['id'], 'selection':'lexicographically first temporal record; not selected by effect',
                         'source':relative,'source_sha256':hashlib.sha256(pdb.read_bytes()).hexdigest(),
                         'projection':'common principal-component projection of input C-alpha coordinates',
                         'projected_coordinates':projected,'h3_indices':record['provenance']['h3_indices']},
        'temporal_units':temporal['inference_units'], 'temporal_aggregate':temporal['aggregate'],
        'ranking':data['ranking']['aggregate'], 'calibration':data['calibration']['development'],
        'model_forward_performed':False,
    }
    return result


def export(fig, out, name):
    for ext in ('png','pdf','svg'):
        kwargs = {'dpi':350,'facecolor':'white'}
        if ext == 'pdf':
            kwargs['metadata']={'CreationDate':None,'ModDate':None}
        if ext == 'svg':
            kwargs['metadata']={'Date':None}
        fig.savefig(out/f'{name}.{ext}',**kwargs)
    plt.close(fig)


def draw_native(data,out):
    fig = plt.figure(figsize=(12.8,7.0))
    fig.text(.045,.952,'Peptide context favors native H3 in 12 of 15 antigen clusters',
             fontsize=18,weight='bold',color=INK)
    fig.text(.045,.907,'ONE-TIME TEMPORAL FINAL  /  31 structures  /  cluster-weighted evidence',
             fontsize=13,color='#637581')
    fig.text(.045,.844,'A',fontsize=14,weight='bold',color=INK)
    fig.text(.075,.844,'Same Fab; remove the peptide',fontsize=14,weight='bold',color=INK)
    coords=data['illustration']['projected_coordinates']
    xy=np.concatenate([np.array(v) for v in coords.values()])
    limits=[xy[:,0].min()-4,xy[:,0].max()+4,xy[:,1].min()-4,xy[:,1].max()+4]
    for x,peptide,label in [(.045,True,'Complex'),(.265,False,'Peptide-stripped')]:
        ax=fig.add_axes([x,.37,.185,.405])
        for chain in ('H','L'):
            if chain in coords:
                p=np.array(coords[chain]);ax.plot(p[:,0],p[:,1],color=GRAY,lw=1.5,zorder=1)
        h=np.array(coords['H'])[data['illustration']['h3_indices']]
        ax.plot(h[:,0],h[:,1],color=TEAL,lw=3,zorder=3)
        if peptide:
            p=np.array(coords['P']);ax.plot(p[:,0],p[:,1],color=GOLD,lw=3,zorder=4)
        ax.axis(limits);ax.set_aspect('equal');ax.axis('off')
        ax.text(.5,-.03,label,transform=ax.transAxes,ha='center',fontsize=14,color=INK,weight='bold')
    fig.text(.238,.58,'\u2192',fontsize=27,color='#8999a1',ha='center')
    handles=[Line2D([0],[0],color=c,lw=3,label=l) for c,l in [(GRAY,'Fab'),(TEAL,'H3'),(GOLD,'Peptide')]]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.245,.287),ncol=3,frameon=False,fontsize=13)
    fig.text(.052,.220,'Native advantage: change in the NLL gap',fontsize=14,weight='bold',color=INK)
    fig.text(.052,.174,'(shuffle - native) with peptide\nminus (shuffle - native) on stripped Fab',
             fontsize=14,color='#4e626e',linespacing=1.4,va='top')
    fig.text(.052,.046,'C-alpha projection: '+data['illustration']['record_id'].replace('pdb_0000','')+
             ' (selected by identifier)',fontsize=11,color='#6b7d86')
    fig.text(.545,.844,'B',fontsize=14,weight='bold',color=INK)
    fig.text(.575,.844,'All 15 clusters, including the negatives',fontsize=14,weight='bold',color=INK)
    ax=fig.add_axes([.615,.303,.335,.465])
    units=sorted(data['temporal_units'],key=lambda r:r['mean_ecls_advantage'],reverse=True)
    values=np.array([r['mean_ecls_advantage'] for r in units])
    colors=[TEAL if v>0 else RUST for v in values]
    ax.axvline(0,color='#85969e',lw=1)
    ax.hlines(range(len(values)),0,values,color=colors,lw=2.2)
    ax.scatter(values,range(len(values)),s=35,c=colors,zorder=3)
    ax.set_yticks(range(len(values)),[r['unit'] for r in units],fontsize=13)
    ax.invert_yaxis();ax.set_xlim(min(-.28,values.min()-.07),max(.7,values.max()+.07))
    ax.tick_params(axis='y',length=0,pad=8);ax.grid(axis='x',alpha=.13)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='x',labelsize=13)
    ax.set_xlabel('Native advantage (nats per residue)',fontsize=13,labelpad=8)
    ax.text(.02,1.035,'Favors shuffle',transform=ax.transAxes,fontsize=13,color=RUST)
    ax.text(.60,1.035,'Favors native',transform=ax.transAxes,fontsize=13,color=TEAL)
    agg=data['temporal_aggregate'];ci=agg['ecls_advantage_mean_ci95']
    mean_ax=fig.add_axes([.615,.090,.335,.052])
    mean_ax.set_xlim(ax.get_xlim());mean_ax.set_ylim(-1,1)
    mean_ax.axvline(0,color='#85969e',lw=.8)
    mean_ax.errorbar(agg['mean_ecls_advantage'],0,xerr=[[agg['mean_ecls_advantage']-ci[0]],[ci[1]-agg['mean_ecls_advantage']]],
                     fmt='D',color=INK,elinewidth=3,capsize=5,markersize=7)
    mean_ax.axis('off')
    fig.text(.615,.161,f"Mean +{agg['mean_ecls_advantage']:.3f}   |   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
             fontsize=13,weight='bold',color=INK)
    fig.text(.615,.046,'Dots: cluster means. Interval: cluster bootstrap.',fontsize=11,color='#6b7d86')
    export(fig,out,'ecls_study_design')


def draw_ranking(data,out):
    fig=plt.figure(figsize=(12.8,7.9))
    fig.text(.05,.95,'A native-recognition signal is not a universal ranking gain',fontsize=18,weight='bold',color=INK)
    fig.text(.05,.908,'DEVELOPMENT PANEL  /  seven antigen clusters  /  same pools under each score',fontsize=13,color='#637581')
    fig.text(.05,.852,'Normalized native rank (higher is better)',fontsize=13,color=INK)
    names=[('bfn','BFN','#496a91'),('esm_if','ESM-IF','#8b709f'),('proteinmpnn','ProteinMPNN',TEAL)]
    calibrated={r['unit']:r for r in data['calibration']['units']}
    for index,(key,title,col) in enumerate(names):
        ax=fig.add_axes([.083+index*.309,.41,.233,.373])
        rows=data['ranking']['generators'][key]['units']
        matrix=np.array([[r['complex_nll_nnr'],r['ecls_nnr'],calibrated[r['unit']][key]] for r in rows])
        for values in matrix:
            ax.plot([0,1,2],values,color=col,alpha=.28,lw=1.1,marker='o',ms=3)
        ax.plot([0,1,2],matrix.mean(axis=0),color=col,lw=3.2,marker='o',ms=7,zorder=4)
        ax.axhline(.5,color='#6f7b82',ls=(0,(3,3)),lw=.9,zorder=0)
        ax.set_ylim(-.06,1.12);ax.set_xlim(-.16,2.16)
        ax.set_xticks([0,1,2],['Complex\nNLL','ECLS','Calibrated*'],fontsize=13)
        ax.set_yticks([0,.5,1],['0','.5','1'])
        ax.tick_params(axis='both',length=0,pad=7,labelsize=13)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(title,loc='left',fontsize=16,weight='bold',color=col,pad=18)
        ax.text(.0,1.025,f"ECLS mean change {matrix[:,1].mean()-matrix[:,0].mean():+.2f}",
                 transform=ax.transAxes,fontsize=13,color=col)
    fig.text(.05,.321,'Thin lines: cluster means over seeds. Thick: generator mean. Coincident values overlap.',fontsize=13,color='#637581')
    fig.text(.05,.277,'Both gain intervals include zero',fontsize=16,weight='bold',color=INK)
    ax=fig.add_axes([.39,.105,.36,.14])
    r=data['ranking'];c=data['calibration']
    for i,(mean,ci,color) in enumerate([(r['overall_mean_gain'],r['overall_gain_ci95'],INK),
                                      (c['mean_gain_over_complex_nll'],c['gain_over_complex_nll_ci95'],GOLD)]):
        ax.errorbar(mean,i,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',color=color,capsize=4,lw=2.2,ms=6)
        ax.text(.52,i,f"{mean:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]",va='center',fontsize=12,color=color)
    ax.set_yticks([0,1],['Universal ECLS','Calibrated*'],fontsize=13);ax.invert_yaxis()
    ax.set_ylim(1.5,-.5);ax.set_xlim(-.25,.42);ax.axvline(0,color='#89959b',lw=.8,ls='--')
    ax.set_xticks([-.2,0,.2,.4]);ax.tick_params(length=0,labelsize=13);ax.spines[['top','right','left']].set_visible(False)
    ax.set_xlabel('Gain over complex NLL (95% cluster-bootstrap CI)',fontsize=12)
    fig.text(.05,.026,'*Exploratory calibration on exposed data; above-random retrieval does not establish improvement over complex NLL.',fontsize=11.5,color='#637581')
    export(fig,out,'ecls_frozen_evidence')


def build(root=ROOT, archive_root=None):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'svg.fonttype':'none','svg.hashsalt':'ecls-narrative-v2','axes.unicode_minus':False})
    if archive_root:
        archive_root=Path(archive_root)
        data=read(archive_root/'figure_source_data.json')
        out=archive_root/'manuscript/figures'
    else:
        root=Path(root);data=collect(root);out=root/'publication/figures'
        path=root/'release/ecls_v1/figure_source_data.json'
        path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    out.mkdir(parents=True,exist_ok=True)
    draw_native(data,out);draw_ranking(data,out)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive-root',type=Path)
    args=p.parse_args();build(archive_root=args.archive_root)
