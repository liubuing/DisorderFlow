"""DisorderFlow web app package (split from app.py, 2026-09-27)."""
# 包结构（均由原 app.py 按职责拆分而来，函数体保持原样）：
#   state      全局状态、设备/配置、检查点加载
#   sequtils   FASTA/序列输入解析与 PDB 生成
#   design     BFN 设计与置信度评估
#   baselines  ProteinMPNN / ESM-IF 基线打分
#   af2        AlphaFold 预测入口
#   pipelines  批量与统一设计管线
#   ui         Gradio 界面组装
# 对外统一出口见 webapp.facade。
