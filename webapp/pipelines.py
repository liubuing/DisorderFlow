"""app.py split: pipelines (auto-generated 2026-09-27, bodies verbatim)."""
#!/usr/bin/env python
"""
蛋白质序列设计平台 — Gradio Web 界面（管理增强版）

访问: http://127.0.0.1:7860  （仅本机访问，外部无法连接）
管理: python manage.py start|stop|restart|status|batch|config|test
"""
import os, json, subprocess, tempfile, re, sys, yaml, time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import gradio as gr
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))
import target_design_helpers as tdh
import cascade_filter as cf
import misfolding_knowledge_base as mkb
import misfolding_pipeline as mfp
import closed_loop_scorer as cls
import closed_loop_orchestrator as clo
from runtime_environment import build_colabfold_command, colabfold_environment

from webapp.baselines import run_esmif, run_mpnn
from webapp.design import build_results_dataframe, run_bfn_antibody, run_bfn_confidence_evaluation, run_bfn_protein
from webapp.sequtils import _save_fasta, detect_chains
from webapp.state import CONFIDENCE_DEFAULTS, DEVICE, load_app_config


def run_batch(pdb_files, tool, region_spec, chains_spec, temperature, num_samples, progress=gr.Progress()):
    if not pdb_files:
        return "请上传 PDB 文件", None, None

    results = []
    all_results_lists = []
    total = len(pdb_files)
    output_dir = Path('batch_results')
    output_dir.mkdir(exist_ok=True)

    for i, pf in enumerate(pdb_files):
        pdb_name = Path(pf.name).stem
        progress((i+1)/total, desc=f"处理 {pdb_name}...")

        try:
            if tool == "BFN (通用蛋白设计)":
                text, fasta_str, results_list = run_bfn_protein(pf.name, region_spec, num_samples, False, True)
            elif tool == "ProteinMPNN":
                text, fasta_str, results_list = run_mpnn(pf.name, chains_spec, num_samples, temperature, 42, "")
            elif tool == "ESM-IF":
                text, fasta_str, results_list = run_esmif(pf.name, chains_spec, float(temperature), num_samples)
            else:
                text = f"未知工具: {tool}"
                fasta_str = ""
                results_list = []
            results.append({'pdb': pdb_name, 'status': 'OK', 'text': text, 'fasta': fasta_str})
            all_results_lists.extend(results_list)
        except Exception as e:
            results.append({'pdb': pdb_name, 'status': 'ERROR', 'error': str(e)})

    # Build summary
    lines = [f"批量处理完成: {total} 个文件\n"]
    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
    lines.append(f"成功: {ok}  失败: {err}\n")
    lines.append("=" * 60)

    for r in results:
        lines.append(f"\n{'─'*60}")
        lines.append(f"📁 {r['pdb']}  [{r['status']}]")
        if r['status'] == 'OK':
            lines.append(r['text'])
        else:
            lines.append(f"错误: {r['error']}")

    # Save summary
    summary = {
        'tool': tool, 'total': total, 'ok': ok, 'error': err,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [{k: v for k, v in r.items() if k not in ('text', 'fasta')} for r in results],
    }
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f'batch_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Also save individual results
    for r in results:
        if r['status'] == 'OK':
            txt_path = output_dir / f"{r['pdb']}_{tool}.txt"
            with open(txt_path, 'w') as f:
                f.write(r['text'])

    lines.append(f"\n结果已保存到: {output_dir}/")

    df = build_results_dataframe(all_results_lists)
    return '\n'.join(lines), str(json_path), df
def run_target_design(target_pdb, target_chain, epitope_region,
                       antibody_pdb, ab_heavy, ab_light,
                       design_tool, constraint_mode, constraint_cutoff,
                       bfn_samples, bfn_stochastic, bfn_eval,
                       mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
                       esmif_temp, esmif_samples):
    """Target-aware constrained design pipeline."""
    if not target_pdb or not os.path.exists(str(target_pdb)):
        return "请先完成步骤1：上传或预测靶点结构", "", "", None

    if not epitope_region or not epitope_region.strip():
        return "请先完成步骤2：分析并选择表位区域", "", "", None

    if not antibody_pdb or not os.path.exists(str(antibody_pdb)):
        return "请先完成步骤3：提供抗体结构", "", "", None

    # Parse epitope region
    epitope_residues = tdh.parse_region_spec(epitope_region)
    if not epitope_residues:
        return f"表位区域格式错误: {epitope_region}", "", "", None

    # Flatten all epitope resseqs
    all_epitope_resseqs = []
    for rlist in epitope_residues.values():
        all_epitope_resseqs.extend(rlist)
    all_epitope_resseqs = sorted(set(all_epitope_resseqs))

    design_lines = []
    validation_text = ""

    try:
        complex_chains = None
        if design_tool == "BFN (抗体CDR设计)":
            complex_chains = detect_chains(antibody_pdb)
            if target_chain not in complex_chains:
                return (
                    "BFN靶点设计要求抗体PDB本身包含已配准的抗原链；"
                    f"当前复合物缺少链 {target_chain}。不能合并两个独立坐标系的PDB。",
                    "", "", None)
        # If constraint mode is on, find antibody residues facing epitope
        design_region = None
        if constraint_mode:
            facing = tdh.find_residues_facing_region(
                antibody_pdb, target_chain, all_epitope_resseqs,
                ab_heavy, constraint_cutoff
            )
            # Also check light chain if specified
            facing_l = []
            if ab_light and ab_light.strip():
                facing_l = tdh.find_residues_facing_region(
                    antibody_pdb, target_chain, all_epitope_resseqs,
                    ab_light, constraint_cutoff
                )
            design_lines.append(f"距离约束模式 (cutoff={constraint_cutoff}Å):")
            design_lines.append(f"  面向表位的重链残基: {len(facing)} 个")
            if ab_light and ab_light.strip():
                design_lines.append(f"  面向表位的轻链残基: {len(facing_l)} 个")

            if not facing and not facing_l:
                return ("未找到面向表位的抗体残基 — 请检查距离阈值或表位/抗体结构是否正确",
                        "", "", None)

            # Build region spec from facing residues
            region_parts = []
            if facing:
                region_parts.append(tdh.residues_to_region_spec(ab_heavy, facing))
            if facing_l:
                region_parts.append(tdh.residues_to_region_spec(ab_light, facing_l))
            design_region = ' '.join(region_parts)
            design_lines.append(f"  设计区域: {design_region}")
            design_lines.append("")
        else:
            # Without constraint, use CDR defaults
            design_region = f"{ab_heavy}:95-102"  # CDR H3 range
            if ab_light and ab_light.strip():
                design_region += f" {ab_light}:89-97"  # CDR L3 range
            design_lines.append(f"非约束模式 — 默认CDR区域: {design_region}")
            design_lines.append("")

        # Run selected design tool
        results_list = []
        if design_tool == "BFN (抗体CDR设计)":
            from modules.bfn_loader import run_bfn_design

            context_chains = [
                chain for chain in complex_chains
                if chain not in {ab_heavy, ab_light}
            ]
            if ab_light and ab_light in complex_chains:
                context_chains.insert(0, ab_light)
            results_list = run_bfn_design(
                antibody_pdb,
                design_region,
                num_samples=int(bfn_samples),
                stochastic=bool(bfn_stochastic),
                context_chains=context_chains,
                antigen_chains=[target_chain],
                device=DEVICE,
            )
            text = "\n".join(
                f"#{index + 1}: {row['sequence']} | PPL={row['ppl']:.2f} | "
                f"ipTM={row['iptm']:.3f}"
                for index, row in enumerate(results_list)
            )
            fasta_str = "\n".join(
                f">BFN_target_sample_{index + 1}\n{row['sequence']}"
                for index, row in enumerate(results_list)
            )
        elif design_tool == "ProteinMPNN":
            ab_chains = ab_heavy
            if ab_light and ab_light.strip():
                ab_chains += " " + ab_light
            text, fasta_str, results_list = run_mpnn(antibody_pdb, ab_chains,
                                       int(mpnn_samples), mpnn_temp, int(mpnn_seed), mpnn_omit)
        elif design_tool == "ESM-IF":
            text, fasta_str, results_list = run_esmif(antibody_pdb, ab_heavy,
                                        float(esmif_temp), int(esmif_samples))
        else:
            return f"未知工具: {design_tool}", "", "", None

        design_lines.append(text)

        # Validation: contact analysis between designed antibody and target
        try:
            contacts, ia, ib, summary = tdh.analyze_contacts(
                antibody_pdb, ab_heavy, target_chain, distance_cutoff=8.0
            )
            if contacts:
                metrics = tdh.score_interface_properties(
                    antibody_pdb, ab_heavy, target_chain, ia, ib
                )
                validation_text = tdh.format_contact_summary(
                    contacts, ia, ib, summary, ab_heavy, target_chain
                )
                validation_text += "\n\n" + tdh.format_interface_report(metrics, summary)
            else:
                validation_text = f"未检测到 {ab_heavy}-{target_chain} 之间的接触 (距离阈值 8Å)"
        except Exception as val_err:
            validation_text = f"验证跳过: {val_err}"

    except Exception as e:
        import traceback
        return f"靶点设计错误:\n{traceback.format_exc()}", "", "", None

    results_df = build_results_dataframe(results_list)
    return '\n'.join(design_lines), fasta_str, validation_text, results_df
def run_design(pdb_path, tool,
               bfn_ab_heavy, bfn_ab_light, bfn_ab_samples, bfn_ab_stochastic, bfn_ab_eval,
               bfn_pro_region, bfn_pro_samples, bfn_pro_stochastic, bfn_pro_eval,
               mpnn_chains, mpnn_temp, mpnn_samples, mpnn_seed, mpnn_omit,
               esmif_chain, esmif_temp, esmif_samples):
    if not pdb_path or not os.path.exists(str(pdb_path)):
        return "请先上传 PDB 文件", None, None, gr.update(visible=False)
    try:
        results_list = []
        if tool == "BFN (抗体CDR设计)":
            text, fasta_str, results_list = run_bfn_antibody(pdb_path, bfn_ab_heavy, bfn_ab_light,
                                    int(bfn_ab_samples), bool(bfn_ab_stochastic), bool(bfn_ab_eval))
        elif tool == "BFN (通用蛋白设计)":
            text, fasta_str, results_list = run_bfn_protein(pdb_path, bfn_pro_region,
                                   int(bfn_pro_samples), bool(bfn_pro_stochastic), bool(bfn_pro_eval))
        elif tool == "ProteinMPNN":
            text, fasta_str, results_list = run_mpnn(pdb_path, mpnn_chains, int(mpnn_samples), mpnn_temp, int(mpnn_seed), mpnn_omit)
        elif tool == "ESM-IF":
            text, fasta_str, results_list = run_esmif(pdb_path, esmif_chain, float(esmif_temp), int(esmif_samples))
        else:
            return f"未知工具: {tool}", None, None, gr.update(visible=False)
    except Exception as e:
        import traceback
        return f"运行错误:\n{traceback.format_exc()}", None, None, gr.update(visible=False)

    # Write FASTA to temp file for download
    fasta_file = None
    if fasta_str:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False,
                                          encoding='utf-8', prefix='design_')
        tmp.write(fasta_str)
        tmp.close()
        fasta_file = tmp.name

    df = build_results_dataframe(results_list)
    return text, fasta_file, df, gr.update(visible=bool(fasta_str))
def run_unified_pipeline(pdb_path, region_spec, num_samples, stochastic, enable_af2, af2_num_recycle, progress=gr.Progress()):
    if not pdb_path or not os.path.exists(str(pdb_path)):
        return "Please upload a PDB file first", pd.DataFrame(), "Pipeline aborted"

    # Stage 0: Pre-design confidence evaluation
    progress(0.0, desc="Stage 0/4: BFN Pre-Design Confidence...")
    conf_text, conf_df = run_bfn_confidence_evaluation(pdb_path, region_spec)
    # Parse mean pLDDT from the report text
    import re as _re
    pre_plddt_match = _re.search(r'pLDDT \(region mean\):\s+([\d.]+)', conf_text)
    pre_iptm_match = _re.search(r'ipTM \(global\):\s+([\d.]+)', conf_text)
    pre_plddt = float(pre_plddt_match.group(1)) if pre_plddt_match else None
    pre_iptm = float(pre_iptm_match.group(1)) if pre_iptm_match else None

    progress(0.15, desc="Stage 1/4: BFN Design...")
    text_design, fasta_str, results_list = run_bfn_protein(
        pdb_path, region_spec, int(num_samples), bool(stochastic), eval_mode=False)

    if not results_list:
        return text_design, pd.DataFrame(), fasta_str or "Design produced no results"

    progress(0.5, desc="Stage 2/4: Cascade Filter...")
    # Pass confidence thresholds from config
    cfg = load_app_config()
    thresholds = cfg.get('confidence', CONFIDENCE_DEFAULTS)
    filter_thresholds = {
        'plddt_min': thresholds.get('plddt_medium', 0.60),
        'iptm_min': thresholds.get('iptm_medium', 0.40),
        'ppl_max': 100.0,
        'entropy_max': 2.5,
    }
    filtered, filter_report = cf.apply_cascade(results_list, thresholds=filter_thresholds)
    if not filtered:
        df = build_results_dataframe(results_list)
        report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: Cascade Filter\n{'='*50}\n{filter_report}"
        return report, df, fasta_str or ""

    progress(0.7, desc="Stage 3/4: Optional AF2 Validation...")
    if enable_af2:
        top_n = min(5, len(filtered))
        top_sequences = [f['sequence'] for f in filtered[:top_n]]
        from af2_validator import validate_sequences
        try:
            af2_results = validate_sequences(top_sequences, output_dir='alphafold_results/pipeline',
                                             num_recycle=int(af2_num_recycle))
            progress(0.9, desc="AF2 validation complete, re-ranking...")
            from cascade_filter import apply_cascade_af2
            af2_filtered, af2_report = apply_cascade_af2(af2_results)
            if af2_filtered:
                df = build_results_dataframe(af2_filtered)
                report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: BFN Cascade Filter (PPL+entropy+pLDDT+ipTM)\n{'='*50}\n{filter_report}\n\n{'='*50}\nStage 3: AF2 Validation + Re-rank\n{'='*50}\n{af2_report}"
                return report, df, fasta_str or ""
        except Exception as e:
            filter_report += f"\n\nAF2 Validation failed: {e}"

    df = build_results_dataframe(filtered)
    report = f"{'='*50}\nStage 0: Pre-Design Confidence\n{'='*50}\n{conf_text}\n\n{text_design}\n\n{'='*50}\nStage 2: Cascade Filter (PPL+entropy+pLDDT+ipTM)\n{'='*50}\n{filter_report}"
    return report, df, fasta_str or ""
def run_unified_pipeline_for_ui(*args, **kwargs):
    """Adapt FASTA text from the pipeline to a Gradio File path."""
    report, dataframe, fasta_text = run_unified_pipeline(*args, **kwargs)
    fasta_path = _save_fasta(fasta_text) if fasta_text and '>' in fasta_text else None
    return report, dataframe, fasta_path
