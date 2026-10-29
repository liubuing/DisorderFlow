"""Aggregated public surface of the old app.py (auto-generated)."""
# 本文件是 2026-09-27 拆分 app.py 时的兼容层：旧代码 `from app import X` 仍可工作，
# 因为这里把各子模块的公开符号全部重导出到包级别。语义上只是 re-export，
# 不含任何业务逻辑；新增功能应放进对应子模块而不是这里。

# 全局状态与配置：设备选择、默认模型配置、BFN/ESM-IF 检查点加载、系统状态汇总
from webapp.state import AA_LETTERS, CONFIDENCE_DEFAULTS, CONFIG_FILE, DEFAULT_MODEL_CONFIG, DEVICE, PROJECT_DIR, get_bfn_ckpt, get_config_display, get_system_status, load_app_config, load_bfn, load_esmif, save_config_from_text
# 序列/结构输入处理：FASTA 解析、链检测、由序列生成 PDB、上传处理
from webapp.sequtils import BB_ATOMS, detect_chains, fasta_to_pdb_file, generate_pdb_from_sequence, get_chain_info, on_fasta_input, on_upload, parse_fasta
# 设计执行：BFN 抗体/蛋白设计、置信度评估、结果诊断与 DataFrame 构建
from webapp.design import build_results_dataframe, confidence_quality_label, diagnose_design_results, run_bfn_antibody, run_bfn_confidence_evaluation, run_bfn_protein
# 基线打分器：ProteinMPNN（run_mpnn）与 ESM-IF（run_esmif）似然
from webapp.baselines import AA_NAMES, run_esmif, run_mpnn
# AlphaFold 预测入口（单条/批量结构预测与 PAE 输出）
from webapp.af2 import run_alphafold_prediction
# 高层管线：单次设计、批量运行、靶点设计、统一管线（含 Gradio UI 适配版）
from webapp.pipelines import run_batch, run_design, run_target_design, run_unified_pipeline, run_unified_pipeline_for_ui
# UI 组装：create_ui 把上述管线挂成 Gradio 界面
from webapp.ui import create_ui
