"""DisorderFlow web package.

后端模块（与界面无关，可被 CLI / 测试直接调用）：
  state      全局状态、设备/配置、检查点加载
  sequtils   FASTA/序列输入解析与 PDB 生成
  design     BFN 设计与置信度评估
  baselines  ProteinMPNN / ESM-IF 基线打分
  af2        AlphaFold 预测入口
  pipelines  批量 / 靶点 / 统一设计管线
  ops        操作层：把一次请求收敛成 OpResult（供 Web 服务调用）
  runs       运行记录与 manifest

界面：webapp/web/（FastAPI + 静态资源），入口 app_web.py。
"""
