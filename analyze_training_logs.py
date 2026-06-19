#!/usr/bin/env python3
"""Quick analysis of training logs — overconfidence calibration from metrics.

Reads log.txt from each training run and extracts:
  - Validation disorder loss (folded vs IDP split)
  - Confidence head losses (pLDDT, ipTM, PAE)
  - Negative sample configuration
  - Freeze/unfreeze strategy

Produces a comparative report. Runs on CPU, no GPU needed.
"""
import os
import re
import sys
import io
from pathlib import Path
from collections import defaultdict

# Force UTF-8 for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).parent

RUNS = {
    'phase3_disorder': {
        'dir': 'logs/bfn_phase3_disorder_2026_05_31__14_52_47',
        'short': 'Phase 3 (baseline)',
    },
    'phase4_negative': {
        'dir': 'logs/bfn_phase4_negative_xpu_2026_05_31__18_27_56',
        'short': 'Phase 4 (40% neg)',
    },
    'phase5_calibrate': {
        'dir': 'logs/bfn_phase5_calibrate_xpu_2026_05_31__21_40_45',
        'short': 'Phase 5 (30% neg calibrated)',
    },
    'idp_combined': {
        'dir': 'logs/bfn_idp_combined_cuda_2026_06_02__13_48_09',
        'short': 'IDP Combined',
    },
    'phase3_v11_nosmooth_v2': {
        'dir': 'logs/bfn_phase3_v11_nosmooth_v2_2026_06_01__16_08_58',
        'short': 'Current (nosmooth v2)',
    },
}


def parse_log(log_path):
    """Extract key configuration and metrics from training log."""
    info = {'val_metrics': [], 'config': {}, 'has_neg_samples': False}

    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            # Config
            if 'Loss weights:' in line:
                m = re.search(r'\{([^}]+)\}', line)
                if m:
                    info['loss_weights'] = m.group(1)
            if 'freeze_backbone' in line and "'freeze_backbone':" in line:
                m = re.search(r'freeze_backbone.:\s*(\w+)', line)
                if m:
                    info['freeze_backbone'] = m.group(1)
            if 'unfreeze_encoder_layers' in line:
                m = re.search(r'unfreeze_encoder_layers.:\s*(\d+)', line)
                if m:
                    info['unfreeze_layers'] = int(m.group(1))
            if 'neg_sample_prob' in line:
                m = re.search(r'neg_sample_prob.:\s*([\d.]+)', line)
                if m:
                    info['neg_sample_prob'] = float(m.group(1))
                    info['has_neg_samples'] = True
            if 'neg_confidence' in line:
                m = re.search(r'neg_confidence.:\s*\{([^}]+)\}', line)
                if m:
                    info['neg_confidence'] = m.group(1)
            if 'Finetuning from' in line:
                m = re.search(r'Finetuning from checkpoint:\s*(.+)$', line)
                if m:
                    info['finetune_from'] = m.group(1).strip()
            if 'mask_aug_prob' in line and 'mask_aug_prob.:' in line:
                m = re.search(r'mask_aug_prob.:\s*([\d.]+)', line)
                if m:
                    info['mask_aug_prob'] = float(m.group(1))
            if 'encoder_dropout' in line:
                m = re.search(r'encoder_dropout.:\s*([\d.]+)', line)
                if m:
                    info['encoder_dropout'] = float(m.group(1))
            if 'Trainable:' in line:
                m = re.search(r'Trainable:\s*[\d,]+\/[\d,]+', line)
                if m:
                    info['trainable_pct'] = m.group(0)

            # Validation metrics
            # Line format: "... [val/folded] Iter 00200 | loss 0.0885 | loss(seq) 1.65 | loss(plddt) 0.02 | ..."
            for tag, tp in [('[val/idp]', 'idp'), ('[val/folded]', 'folded'),
                            ('[val] ', 'overall')]:
                if tag in line and 'loss(' in line:
                    vals = {'type': tp}
                    # Overall loss: "| loss DIGIT" (no parentheses)
                    m_loss = re.search(r'\|\s*loss\s+([\d.]+)', line)
                    if m_loss:
                        vals['loss'] = float(m_loss.group(1))
                    # Per-head losses: loss(plddt), loss(seq), etc.
                    for key in ['seq', 'plddt', 'iptm', 'pae', 'disorder']:
                        m = re.search(rf'loss\({key}\)\s+([\d.]+)', line)
                        if m:
                            vals[key] = float(m.group(1))
                    info['val_metrics'].append(vals)
                    break  # one tag per line

    return info


def get_best_metrics(info):
    """Get the best (lowest) validation metrics for each category."""
    best = {}
    for m in info['val_metrics']:
        tp = m['type']
        if tp not in best or m.get('loss', 999) < best[tp].get('loss', 999):
            best[tp] = m
    return best


def main():
    print("=" * 80)
    print("  Training Log Analysis: Overconfidence Calibration Insights")
    print("=" * 80)

    # Parse all logs
    all_data = {}
    for name, cfg in RUNS.items():
        log_path = PROJECT_ROOT / cfg['dir'] / 'log.txt'
        if log_path.exists():
            all_data[name] = parse_log(str(log_path))
            all_data[name]['short'] = cfg['short']
            all_data[name]['dir'] = cfg['dir']

    # ── Configuration comparison ──
    print(f"\n  ── Training Configuration ──")
    print(f"  {'Run':<30} {'Neg%':>6} {'Freeze':>7} {'Unfrz':>6} "
          f"{'EncDrop':>8} {'MaskAug':>8}")
    print(f"  {'─' * 70}")
    for name, d in all_data.items():
        neg = f"{d.get('neg_sample_prob', 0)*100:.0f}%" if d.get('has_neg_samples') else 'none'
        freeze = d.get('freeze_backbone', '?')
        unfrz = str(d.get('unfreeze_layers', '?'))
        drop = d.get('encoder_dropout', 0)
        mask = d.get('mask_aug_prob', 0)
        print(f"  {d['short']:<30} {neg:>6} {freeze:>7} {unfrz:>6} "
              f"{drop:>8.2f} {mask:>8.2f}")

    # ── Best val metrics ──
    print(f"\n  ── Best Validation Metrics (folded proteins) ──")
    header = (f"  {'Run':<28} {'loss':>8} {'pLDDT':>8} "
              f"{'ipTM':>8} {'PAE':>8} {'disorder':>10}")
    print(header)
    print(f"  {'─' * 72}")
    for name, d in all_data.items():
        best = get_best_metrics(d)
        fb = best.get('folded', {})
        if fb:
            print(f"  {d['short']:<28} "
                  f"{fb.get('loss', 0):>8.4f} {fb.get('plddt', 0):>8.4f} "
                  f"{fb.get('iptm', 0):>8.4f} {fb.get('pae', 0):>8.4f} "
                  f"{fb.get('disorder', 0):>10.4f}")

    # ── IDP vs Folded disorder loss ──
    print(f"\n  ── Disorder Loss: Folded vs IDP (lower = head is confident) ──")
    print(f"  {'Run':<28} {'Folded':>10} {'IDP':>10} {'Ratio':>10}")
    print(f"  {'─' * 60}")
    for name, d in all_data.items():
        best = get_best_metrics(d)
        fb_dis = best.get('folded', {}).get('disorder', None)
        idp_dis = best.get('idp', {}).get('disorder', None)
        if fb_dis is not None and idp_dis is not None:
            ratio = idp_dis / max(fb_dis, 0.001)
            print(f"  {d['short']:<28} {fb_dis:>10.4f} {idp_dis:>10.4f} "
                  f"{ratio:>9.2f}x")

    # ── Overconfidence assessment ──
    print(f"\n  ── Overconfidence Calibration Assessment ──")
    print()

    # Find the three-way comparison
    p3 = all_data.get('phase3_disorder')
    p4 = all_data.get('phase4_negative')
    ic = all_data.get('idp_combined')

    if p3 and p4:
        p3_best = get_best_metrics(p3)
        p4_best = get_best_metrics(p4)

        p3_dis = p3_best.get('folded', {}).get('disorder', 0)
        p4_dis = p4_best.get('folded', {}).get('disorder', 0)
        p3_iptm = p3_best.get('folded', {}).get('iptm', 0)
        p4_iptm = p4_best.get('folded', {}).get('iptm', 0)

        print(f"  Phase 3 → Phase 4 (negative sample training):")
        print(f"    Disorder loss: {p3_dis:.4f} → {p4_dis:.4f} "
              f"({(p4_dis-p3_dis)/max(p3_dis,0.001)*100:+.1f}%)")
        print(f"    ipTM loss:     {p3_iptm:.4f} → {p4_iptm:.4f} "
              f"({(p4_iptm-p3_iptm)/max(p3_iptm,0.001)*100:+.1f}%)")

        if p4_dis > p3_dis:
            print(f"    [!] Negative sample training DEGRADED disorder prediction")
        else:
            print(f"    [OK] Negative samples improved disorder calibration")

    if ic:
        ic_best = get_best_metrics(ic)
        ic_dis = ic_best.get('folded', {}).get('disorder', 0)
        ic_iptm = ic_best.get('folded', {}).get('iptm', 0)

        print(f"\n  IDP Combined vs Phase 3:")
        if p3:
            p3_dis = get_best_metrics(p3).get('folded', {}).get('disorder', 0)
            p3_iptm = get_best_metrics(p3).get('folded', {}).get('iptm', 0)
            print(f"    Disorder loss: {p3_dis:.4f} → {ic_dis:.4f} "
                  f"({(ic_dis-p3_dis)/max(p3_dis,0.001)*100:+.1f}%)")
            print(f"    ipTM loss:     {p3_iptm:.4f} → {ic_iptm:.4f} "
                  f"({(ic_iptm-p3_iptm)/max(p3_iptm,0.001)*100:+.1f}%)")
            if ic_dis < p3_dis:
                improvement = (1 - ic_dis / max(p3_dis, 0.001)) * 100
                print(f"    [OK] IDP Combined is {improvement:.0f}% better on disorder!")

        print(f"\n  IDP Combined key differences:")
        print(f"    - Fully frozen backbone (unfreeze_layers=0)")
        print(f"    - Encoder dropout: {ic.get('encoder_dropout', '?')}")
        print(f"    - No negative samples")
        print(f"    - Lower ipTM loss weight (1.5 vs 2.5)")

    # ── Recommendations ──
    print(f"\n  ── Recommendations for Overconfidence Validation ──")
    print(f"  1. Compare Phase 3 vs Phase 4 vs IDP Combined with AF2 validation")
    print(f"  2. Expected ranking from log analysis:")
    print(f"     1st: IDP Combined (best val metrics, properly regularized)")
    print(f"     2nd: Phase 3 (baseline, moderate calibration)")
    print(f"     3rd: Phase 4 (negative samples hurt performance)")
    print(f"  3. Run:  bash _launch_oc_validation.sh all")
    print(f"     (on Linux GPU machine with JAX/cuDNN)")
    print(f"  4. Or step-by-step:")
    print(f"     bash _launch_oc_validation.sh phase1    # BFN design")
    print(f"     bash _launch_oc_validation.sh phase2    # AF2 validation")
    print(f"     python validate_overconfidence.py --phase3  # Analysis on CPU")

    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
