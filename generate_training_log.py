#!/usr/bin/env python
"""Generate training completion log for BFN confidence head fine-tuning."""
import sys, os, json, re, time
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(r'C:\biological\AntibodyDesignBFN-main\AntibodyDesignBFN-main')
LOG_DIR = Path(r'C:\biological\AntibodyDesignBFN-main\日志')
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_training_output(output_path):
    """Parse training output file for key metrics."""
    with open(output_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_iters = []
    val_iters = []
    best_losses = []

    for line in lines:
        # Train: [train] Iter 00020 | loss 0.4911 | loss(seq) 0.0000 | loss(plddt) 0.4546 | loss(iptm) ... | loss(pae) ... | ...
        if '[train] Iter' in line:
            parts = line.split('|')
            iter_str = parts[0].split('Iter')[1].strip()
            it = int(iter_str)
            loss = float(parts[1].split('loss')[1].strip())
            # parts[2] is loss(seq), skip it — confidence training only
            plddt = float(parts[3].split('loss(plddt)')[1].strip()) if len(parts) > 3 and 'plddt' in parts[3] else None
            iptm = float(parts[4].split('loss(iptm)')[1].strip()) if len(parts) > 4 and 'iptm' in parts[4] else None
            pae = float(parts[5].split('loss(pae)')[1].strip()) if len(parts) > 5 and 'pae' in parts[5] else None
            train_iters.append({'it': it, 'loss': loss, 'plddt': plddt, 'iptm': iptm, 'pae': pae})

        # Val: [val] Iter 00100 | loss 0.3759 | loss(seq) ... | loss(plddt) ... | ...
        if '[val] Iter' in line:
            parts = line.split('|')
            iter_str = parts[0].split('Iter')[1].strip()
            it = int(iter_str)
            loss = float(parts[1].split('loss')[1].strip())
            plddt = float(parts[3].split('loss(plddt)')[1].strip()) if len(parts) > 3 and 'plddt' in parts[3] else None
            iptm = float(parts[4].split('loss(iptm)')[1].strip()) if len(parts) > 4 and 'iptm' in parts[4] else None
            pae = float(parts[5].split('loss(pae)')[1].strip()) if len(parts) > 5 and 'pae' in parts[5] else None
            val_iters.append({'it': it, 'loss': loss, 'plddt': plddt, 'iptm': iptm, 'pae': pae})

        # Best model
        if 'New best model saved' in line:
            loss_str = line.split('with loss')[1].strip()
            best_losses.append({'it': val_iters[-1]['it'] if val_iters else 0, 'loss': float(loss_str)})

    return train_iters, val_iters, best_losses


def format_markdown(train_iters, val_iters, best_losses,
                    eval_results=None, log_dir_name='', total_iters=20000):
    """Generate markdown training log."""

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Find key milestones
    milestones = {}
    for v in val_iters:
        if v['it'] in [100, 500, 1000, 5000, 10000, 15000, 20000] or v == val_iters[-1]:
            milestones[v['it']] = v

    # Find final metrics
    final_train = train_iters[-1] if train_iters else None
    final_val = val_iters[-1] if val_iters else None
    best_val = best_losses[-1] if best_losses else None

    md = f"""# BFN置信度头微调训练日志

**生成时间:** {now}
**项目:** AntibodyDesignBFN — Confidence Head Fine-Tuning
**日志目录:** {log_dir_name}

---

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| 基础检查点 | YueHuLab/AntibodyDesignBFN best.pt (113MB) |
| 训练数据集 | 234蛋白质 (187 train / 47 val) |
| 数据来源 | EBI AFDB v6 + PDB + ColabFold |
| 批次大小 | 1 |
| 最大迭代次数 | {total_iters} |
| 学习率 | 1e-4 |
| 调度器 | Plateau (factor=0.5, patience=30) |
| 预热步数 | 200 |
| 损失权重 | pLDDT=1.0, ipTM=1.0, PAE=0.3 |
| 冻结骨干 | True (133,635 / 10,200,662 = 1.3%) |
| AMP | 禁用 (float16不兼容) |
| 设备 | CUDA (NVIDIA RTX 5060 Laptop) |

## 2. 关键修复（本次训练前应用）

| 修复 | 描述 |
|------|------|
| PAE尺度匹配 | receiver PAE输出从 softplus*10 改为 sigmoid [0,1]，与目标 af2_pae/31.0 对齐 |
| 初始PAE损失改善 | ~90,000倍 (36.6 → 0.0004) |

## 3. 训练进度
"""

    # Milestone table
    md += f"""
| 迭代 | 训练损失 | pLDDT | ipTM | PAE | 验证损失 | pLDDT | ipTM | PAE |
|------|---------|-------|------|-----|---------|-------|------|-----|
"""

    for it in sorted(milestones.keys()):
        v = milestones[it]
        # Find closest train iter
        train_match = None
        for t in train_iters:
            if t['it'] >= it:
                break
            train_match = t
        if train_match:
            md += f"| {v['it']} | {train_match['loss']:.4f} | {train_match['plddt']:.4f} | {train_match['iptm']:.4f} | {train_match['pae']:.4f} | {v['loss']:.4f} | {v['plddt']:.4f} | {v['iptm']:.4f} | {v['pae']:.4f} |\n"

    md += f"""
## 4. 验证损失曲线
"""
    if best_losses:
        md += f"""
| 最佳 | 迭代 | 损失 |
|------|------|------|
"""
        for b in best_losses:
            md += f"| ★ | {b['it']} | {b['loss']:.4f} |\n"

    # Final metrics
    if final_train:
        val_loss_str = f"{final_val['loss']:.4f}" if final_val else 'N/A'
        val_plddt_str = f"{final_val['plddt']:.4f}" if final_val else 'N/A'
        val_iptm_str = f"{final_val['iptm']:.4f}" if final_val else 'N/A'
        val_pae_str = f"{final_val['pae']:.4f}" if final_val else 'N/A'
        best_str = f"{best_val['loss']:.4f} (iter {best_val['it']})" if best_val else 'N/A'
        md += f"""
## 5. 最终指标

| 指标 | 训练 | 验证 |
|------|------|------|
| 总损失 | {final_train['loss']:.4f} | {val_loss_str} |
| pLDDT | {final_train['plddt']:.4f} | {val_plddt_str} |
| ipTM | {final_train['iptm']:.4f} | {val_iptm_str} |
| PAE | {final_train['pae']:.4f} | {val_pae_str} |
| 最佳验证损失 | — | {best_str} |
"""

    # Evaluation results
    if eval_results:
        md += f"""
## 6. 评估结果 (验证集)

| 指标 | 值 |
|------|-----|
| pLDDT Pearson r | {eval_results.get('plddt_pearson_r', 'N/A')} |
| pLDDT Spearman ρ | {eval_results.get('plddt_spearman_rho', 'N/A')} |
| pLDDT MAE | {eval_results.get('plddt_mae', 'N/A')} |
| ipTM Pearson r | {eval_results.get('iptm_pearson_r', 'N/A')} |
| ipTM Spearman ρ | {eval_results.get('iptm_spearman_rho', 'N/A')} |
| ipTM MAE | {eval_results.get('iptm_mae', 'N/A')} |
"""

    md += f"""
## 7. 与上次训练对比

| 指标 | 上次 (修复前) | 本次 (修复后) |
|------|-------------|-------------|
| 初始PAE损失 | 74.30 | 0.118 |
| 初始总损失 | 22.59 | 0.491 |
"""

    if final_val:
        md += f"| 最终验证损失 | — | {final_val['loss']:.4f} |\n"

    md += """
## 8. 已知问题

1. **IDP盲区**: 训练数据缺乏固有无序蛋白，对Tau等IDP的预测偏高（pLDDT r≈0.21）
2. **长度限制**: 训练数据L=50-250，无法泛化到更长蛋白
3. **编码器冻结**: 仅在冻结特征上训练简单MLP，上限受限

---
*由 generate_training_log.py 自动生成*
"""
    return md


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output_file', required=True, help='Training output text file')
    p.add_argument('--eval_json', default=None, help='Evaluation results JSON')
    p.add_argument('--log_dir_name', default='', help='Training log directory name')
    args = p.parse_args()

    train_iters, val_iters, best_losses = parse_training_output(args.output_file)

    eval_results = None
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_data = json.load(f)
            eval_results = eval_data.get('global', {})

    md = format_markdown(train_iters, val_iters, best_losses, eval_results, args.log_dir_name)

    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    out_path = LOG_DIR / f'training_log_{timestamp}.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f'Training log saved to: {out_path}')
    return out_path


if __name__ == '__main__':
    main()
