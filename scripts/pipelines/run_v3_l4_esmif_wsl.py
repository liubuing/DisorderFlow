"""ESM-IF L4 完整性冒烟测试（WSL 内运行）：天然 CDR 序列的逆折叠对数似然
应显著高于同一骨架上组成打乱的序列——若打乱序列反而得分更高，说明 ESM-IF
评分方向或骨架坐标提取有误，后续依赖它的设计臂不可信。

对 3 个抗淀粉样蛋白-beta 参考抗体的重链 CDR（Chothia H1/H2/H3）做
native vs scrambled 的坐标条件似然对比，逐 PDB 输出 PASS/FAIL。"""
import esm,esm.inverse_folding.util as util,torch,numpy as np,os,random,sys
# 固定工作目录到 WSL 挂载点下的项目根：本脚本由 Windows 侧调度、WSL 内执行，
# 数据路径（data/anti_abeta_refs）均相对项目根。
os.chdir("/mnt/d/biological/DisorderFlow")
# ESM-IF1（GVP 编码器，142M 参数）：从骨架坐标条件预测序列的逆折叠模型
model,alphabet=esm.pretrained.esm_if1_gvp4_t16_142M_UR50();model.eval()
from Bio.PDB import PDBParser
AA3={"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
for pdb_id in ["4HIX","5CSZ","3UOT"]:
    pdb=f"data/anti_abeta_refs/{pdb_id}.pdb"
    if not os.path.exists(pdb): continue
    s=PDBParser(QUIET=True).get_structure("s",pdb)
    cr=[(26,32),(52,56),(95,102)]  # Chothia CDR-H1/H2/H3 编号区间
    crd=[];seq=[]
    # 只取重链 H 上落在 CDR 区间、且 N/CA/C 骨架完整的残基——
    # ESM-IF 的 score_sequence 需要每个残基的 N-CA-C 坐标与对应氨基酸
    for res in s[0]["H"]:
        ri=res.id[1]
        if any(sr<=ri<=er for sr,er in cr):
            if all(a in res for a in ["N","CA","C"]):
                crd.append([res["N"].get_coord(),res["CA"].get_coord(),res["C"].get_coord()])
                seq.append(AA3.get(res.resname.strip(),"X"))
    s_str="".join(seq)
    # 天然序列的（坐标条件）对数似然：ll_full 含序列先验项，ll_coord 只含坐标条件项
    ll_full,ll_coord=util.score_sequence(model,alphabet,np.array(crd),s_str)
    # 组成打乱对照：同一骨架、同一氨基酸组成、只打乱顺序——用于隔离"序列顺序"贡献
    scram="".join(random.sample(list(s_str),len(s_str)))
    ll_s_full,ll_s_coord=util.score_sequence(model,alphabet,np.array(crd),scram)
    sep=ll_coord-ll_s_coord
    print(f"{pdb_id}: native={ll_coord:.4f} scram={ll_s_coord:.4f} sep={sep:+.4f} {'PASS' if sep>0 else 'FAIL'}")
print("ESM-IF L4 native vs scrambled DONE")
