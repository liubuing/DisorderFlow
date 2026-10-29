#!/usr/bin/env python3
"""Build split-scoped, non-circular sequence disorder profile artifacts.

The current clustered Phase3 records do not contain experimental B-factors,
UniProt mappings, or conformational ensembles. This builder therefore uses a
versioned charge-hydropathy heuristic and records that source explicitly. It
must not be described as IUPred output or experimental ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# 20 种标准氨基酸（索引即 LMDB 中 aa 编码 0-19）。
AA = "ACDEFGHIKLMNPQRSTVWY"
# Kyte-Doolittle 疏水标度：值越高越疏水，无序倾向与低疏水/高净电荷相关。
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
# 生理 pH 下的形式电荷近似；组氨酸取部分电荷 0.1（pKa 约 6.0，多数中性略偏正）。
CHARGE = {'R': 1.0, 'K': 1.0, 'D': -1.0, 'E': -1.0, 'H': 0.1}
# 来源契约标识：明确这是启发式弱标签，禁止对外表述为 IUPred 或实验真值。
SOURCE = "charge_hydropathy_heuristic_v1"


def sequence_profile(aa_indices, window=15):
    # 由氨基酸编码序列计算逐残基无序倾向弱标签（值域 0-1，越高越可能无序）。
    # 参数:
    #   aa_indices: 20 以内的整数编码序列；越界索引按甘氨酸（G）兜底处理。
    #   window: 滑窗窗口长度，15 个残基约对应一条中等二级结构元素的尺度，
    #     与 IUPred 类方法的常用窗口量级一致。
    # 返回: float32 数组，长度与输入相同。
    # 实现: 采用 Uversky 式电荷-疏水组合——线性组合 (平均|电荷| - 平均疏水)
    # 乘以斜率 6.0、加偏置 0.5 后过 sigmoid 压到 0-1；序列两端各取
    # min(5, len//4) 个残基乘 0.3→1.0 的线性坡道，抑制边缘窗口统计不稳
    # 造成的虚假高值。注意：这是启发式代理，不是实验测量。
    letters = [AA[int(index)] if 0 <= int(index) < 20 else 'G' for index in aa_indices]
    hydropathy = np.asarray([(KD[aa] + 4.5) / 9.0 for aa in letters], dtype=np.float64)
    charge = np.asarray([CHARGE.get(aa, 0.0) for aa in letters], dtype=np.float64)
    output = np.zeros(len(letters), dtype=np.float64)
    half = window // 2
    for index in range(len(letters)):
        start, end = max(0, index - half), min(len(letters), index + half + 1)
        raw = 6.0 * (np.abs(charge[start:end]).mean() - hydropathy[start:end].mean()) + 0.5
        output[index] = 1.0 / (1.0 + np.exp(-raw))
    edge = min(5, len(output) // 4)
    if edge:
        ramp = np.linspace(0.3, 1.0, edge)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    return output.astype(np.float32)


def ids_digest(ids):
    # 对样本 ID 列表取 SHA-256 摘要，把工件与确切的 ID 顺序绑定（防错配）。
    return hashlib.sha256('\n'.join(map(str, ids)).encode('utf-8')).hexdigest()


def build_artifact(lmdb_path, split, output_path):
    # 按 split 构建无序倾向 profile 工件与配套审计 JSON。
    # 参数:
    #   lmdb_path: Phase3 记录 LMDB 路径；同名 '-ids' 文件提供样本顺序。
    #   split: 'train' 或 'val'，写入工件用于划分作用域，避免跨划分泄漏。
    #   output_path: 输出 pickle 路径；旁路另写一份 .audit.json（不含逐样本
    #     profile/provenance，便于人工审查而不膨胀体积）。
    # 返回: 无（落盘）。provenance 中显式记录 experimental_ground_truth=False。
    # 缺键即抛错（fail-closed），宁可失败也不产出静默缺数据的工件。
    ids_path = Path(f"{lmdb_path}-ids")
    ids = pickle.loads(ids_path.read_bytes())
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    profiles, provenance = {}, {}
    lengths = []
    with env.begin() as transaction:
        for sample_id in ids:
            raw = transaction.get(str(sample_id).encode())
            if raw is None:
                raise RuntimeError(f"LMDB key missing for {sample_id}")
            record = pickle.loads(raw)
            antigen = record.get('antigen')
            if antigen is None or 'aa' not in antigen:
                raise RuntimeError(f"Antigen sequence missing for {sample_id}")
            aa = antigen['aa'].numpy() if hasattr(antigen['aa'], 'numpy') else np.asarray(antigen['aa'])
            profile = sequence_profile(aa)
            profiles[str(sample_id)] = profile
            lengths.append(len(profile))
            provenance[str(sample_id)] = {
                'source': SOURCE,
                'pdb_id': str(record.get('pdb_id', record.get('id', sample_id))),
                'length': len(profile),
                'window': 15,
                'experimental_ground_truth': False,
                'model_generated': False,
            }
    env.close()
    means = np.asarray([profile.mean() for profile in profiles.values()])
    artifact = {
        'schema_version': 1,
        'split': split,
        'lmdb_path': str(Path(lmdb_path).resolve()),
        'ids_path': str(ids_path.resolve()),
        'ids_sha256': ids_digest(ids),
        'source_contract': {
            'primary_source': SOURCE,
            'claim_scope': 'non-circular sequence-derived weak profile',
            'unsupported_claims': ['experimental disorder label', 'IUPred prediction'],
        },
        'profiles': profiles,
        'provenance': provenance,
        'stats': {
            'n_ids': len(ids),
            'n_profiles': len(profiles),
            'length_min': min(lengths),
            'length_max': max(lengths),
            'mean_profile_quantiles': {
                str(q): float(np.quantile(means, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    output_path.with_suffix('.audit.json').write_text(
        json.dumps({key: value for key, value in artifact.items() if key not in ('profiles', 'provenance')}, indent=2),
        encoding='utf-8',
    )
    print(json.dumps({'split': split, **artifact['stats'], 'output': str(output_path)}, indent=2))


def main():
    # 命令行入口：--lmdb/--split/--output 均为必填，防止误用默认路径。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('train', 'val'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_artifact(args.lmdb, args.split, args.output)


if __name__ == '__main__':
    main()
