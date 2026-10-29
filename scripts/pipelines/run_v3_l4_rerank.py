"""L4 ESM-IF re-rank: score P3 generated designs with inverse folding likelihood.
KEY DIFFERENCE from L3: ESM-IF scores p(seq|structure) — sequence-DEPENDENT.
Same backbone, different CDR sequence = different score. This is what L3 couldn't do."""
import esm,esm.inverse_folding.util as util,torch,numpy as np,os,json,time,random,sys
os.chdir("/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow")
sys.path.insert(0,"/mnt/d/biological/DisorderFlow/modules")
from Bio.PDB import PDBParser
from scipy.stats import kendalltau

# 加载 ESM-IF1 (142M 参数 GVP 版) 逆折叠模型: 输入骨架坐标, 输出 p(seq|structure),
# 分数越高表示该序列在该骨架上越"可折叠", 以此重排 P3 阶段生成的设计
model,alphabet=esm.pretrained.esm_if1_gvp4_t16_142M_UR50();model.eval()
# 三字母 -> 单字母氨基酸映射, 用于 PDB 残基名与设计序列的互转
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# Load P3 library
# 读取 P3 阶段的全部设计结果, 只取前 100 条进入 L4 重排(控制评分耗时)
with open("idp_design_results/p3_scale_all_20260629_030111.json") as f:
    data=json.load(f)
designs=data.get("designs",data.get("ensemble",[]))[:100]
print(f"L4 ESM-IF re-ranking {len(designs)} generated designs")
print(f"  Key test: can ESM-IF distinguish CDR quality on SAME backbone?")
print("="*60)

# Also add known native CDRs as positive controls
# 阳性对照: 两条已知抗 Aβ 抗体(4HIX/5CSZ)的天然 CDR-H 序列
NATIVES=[
    {"sequence":"GFTFSNYRSGGGYCVRYDHY","epitope":"4HIX_native","scaffold":"4HIX","composite":0,"cdr_spec":"H:26-33,H:51-58,H:95-102","scaffold_chain":"H","scaffold_path":"data/anti_abeta_refs/4HIX.pdb"},
    {"sequence":"GFTFSSYAISGSGGSTYYADSVKG","epitope":"5CSZ_native","scaffold":"5CSZ","composite":0,"cdr_spec":"H:26-33,H:51-58,H:95-102","scaffold_chain":"H","scaffold_path":"data/anti_abeta_refs/5CSZ.pdb"},
]
# Add scrambled versions as negative controls
# 阴性对照: 把天然 CDR 随机打乱(保持氨基酸组成), ESM-IF 打分应显著低于天然序列
for nat in NATIVES[:]:
    s=list(nat["sequence"]);random.shuffle(s)
    scram={**nat,"sequence":"".join(s),"epitope":nat["epitope"]+"_scram","composite":-1}
    NATIVES.append(scram)
designs=NATIVES+designs
print(f"Added {len(NATIVES)} controls (native+scrambled)")

t0=time.time()
# 主循环: 每条设计先嫁接到支架骨架, 再用 ESM-IF 对 CDR 区段打逆折叠对数似然分
scores=[]
for i,d in enumerate(designs):
    sp=d.get("scaffold_path","") or f'data/misfolding_targets/{d.get("scaffold","3STB")}.pdb'
    if not os.path.exists(sp):
        sp=f'data/misfolding_targets/{d.get("scaffold","3STB")}.pdb'
    if not os.path.exists(sp):continue
    s=PDBParser(QUIET=True).get_structure("s",sp)
    # 逐条设计的支架链与 CDR 区段可覆盖默认值; 解析 "H:26-33" 为 (起,止) 残基号对(含端点)
    ch=d.get("scaffold_chain","A")
    cs=d.get("cdr_spec","A:26-33,A:51-58,A:97-113")
    cr=[(int(p.split(":")[1].split("-")[0]),int(p.split(":")[1].split("-")[1]))for p in cs.split(",")]
    # Graft CDR
    cdr=d["sequence"];pos=0
    for st,en in cr:
        for j in range(st,en+1):
            if pos<len(cdr):
                try:s[0][ch][j].resname=AA3.get(cdr[pos],"GLY")
                except:pass;pos+=1
    # Extract backbone coords (N,CA,C only)
    # 只取 CDR 区段内同时具备 N/CA/C 原子的残基, 组成 ESM-IF 所需的骨架坐标与对应序列
    crd=[];seq=[]
    for res in s[0][ch]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr):
            if all(a in res for a in ["N","CA","C"]):
                crd.append([res["N"].get_coord(),res["CA"].get_coord(),res["C"].get_coord()])
                aa=AA3.get(res.resname.strip(),"X")
                seq.append(aa)
    s_str="".join(seq)
    if len(crd)<3:continue
    try:
        # ll_full: 整条给定序列的对数似然; ll_coord: 仅坐标条件的似然(去除序列自洽成分)
        ll_full,ll_coord=util.score_sequence(model,alphabet,np.array(crd),s_str)
        d["l4_ll_full"]=float(ll_full);d["l4_ll_coord"]=float(ll_coord)
        scores.append(ll_coord)
    except Exception as e:
        # 评分失败(如坐标异常)时记 -999 哨兵值, 保证流水线不中断
        d["l4_ll_full"]=-999;d["l4_ll_coord"]=-999
        scores.append(-999)
    if(i+1)%20==0:print(f"  {i+1}/{len(designs)} ({time.time()-t0:.0f}s)")

# Report
# 分三组统计: 天然 CDR / 打乱对照 / 生成设计; 并用 Kendall tau 检验 L4 分数
# 与 P3 旧综合分的一致性(tau 接近 0 说明 L4 提供了新信息)
native_scores=[d["l4_ll_coord"] for d in designs if "native" in d.get("epitope","") and "scram" not in d.get("epitope","")]
scram_scores=[d["l4_ll_coord"] for d in designs if "scram" in d.get("epitope","")]
gen_scores=[d["l4_ll_coord"] for d in designs[len(NATIVES):]]
if len(gen_scores)>0:
    old_c=[d.get("composite",0) for d in designs[len(NATIVES):]]
    tau,p=kendalltau(old_c,gen_scores)
else:
    tau,p=0,1

print(f"\n=== L4 ESM-IF Results ===")
if native_scores: print(f"Native CDR mean: {np.mean(native_scores):.4f}")
if scram_scores: print(f"Scrambled mean:  {np.mean(scram_scores):.4f}")
print(f"Generated mean:  {np.mean(gen_scores):.4f}")
print(f"Generated range: {np.min(gen_scores):.4f} - {np.max(gen_scores):.4f}")
if native_scores and scram_scores:
    print(f"Separation (native-scram): {np.mean(native_scores)-np.mean(scram_scores):+.4f}")
print(f"Kendall tau (vs old composite): {tau:.4f} (p={p:.4f})")

# Sort by L4 score
# 按 L4 坐标条件似然降序输出 Top-10, 并检查天然 CDR 是否排在打乱对照之前(方法有效性 sanity check)
designs.sort(key=lambda d:d.get("l4_ll_coord",-999),reverse=True)
print(f"\nTop-10 L4 ESM-IF:")
for i,d in enumerate(designs[:10]):
    tag="[NATIVE]" if "native" in d.get("epitope","") else ("[SCRAM]" if "scram" in d.get("epitope","") else "")
    print(f"#{i+1:2d} {tag} LL={d['l4_ll_coord']:.4f} {d.get('epitope','')[:20]} {d.get('scaffold','')}")

# Check: are native CDRs ranked higher than scrambled?
native_ranks=[i for i,d in enumerate(designs) if "native" in d.get("epitope","") and "scram" not in d.get("epitope","")]
scram_ranks=[i for i,d in enumerate(designs) if "scram" in d.get("epitope","")]
print(f"\nNative CDR ranks: {native_ranks} (top={min(native_ranks) if native_ranks else 'N/A'})")
print(f"Scrambled CDR ranks: {scram_ranks}")

ts=time.strftime("%Y%m%d_%H%M%S")
out=f"idp_design_results/l4_rerank_{ts}.json"
with open(out,"w") as f:json.dump(designs,f,indent=2)
print(f"Saved {out}")
