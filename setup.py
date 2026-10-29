"""Compatibility entry point; package metadata lives in pyproject.toml."""

# 兼容性入口：真实的包元数据（名称、版本、依赖）全部在 pyproject.toml 中声明。
# 这里只调用无参 setup()，让 setuptools 从 pyproject.toml 读取配置——
# 保留此文件是为了兼容仍执行 `python setup.py ...` 的旧工具链。
from setuptools import setup


setup()
