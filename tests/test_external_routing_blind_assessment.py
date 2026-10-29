"""外部路由盲评估入口的冒烟测试：守护评估模块可导入且 main 入口可调用，
防止盲评估脚本因 import 回归而静默失效。"""
from scripts.assess_external_routing_blind import main


# 冒烟断言：盲评估入口存在且可调用，防止重构破坏外部路由评估的启动路径。
def test_assessment_module_imports():
    assert callable(main)
