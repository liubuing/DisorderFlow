"""数据集子包的聚合入口：导入即触发各数据集类的 @register_dataset 注册。

每个子模块按需包裹在 try/except ImportError 中——部分数据集依赖较重的可选库
（lmdb、Biopython 等），缺失时只降级警告而不阻断整个包的导入。
"""
# Eager imports — @register_dataset decorators run at import time
# Wrap heavy-dependency imports with error handling

from .protein import preprocess_protein_structure

# 静默吞掉 ImportError 的分支对应"可选依赖"数据集：
# 只有在实际按名字取用时（get_dataset）才会因未注册而报错
try:
    from .sabdab import SAbDabDataset
except ImportError:
    pass

try:
    from .custom import CustomDataset
except ImportError:
    pass

try:
    from .lmdb_dataset import LMDBDataset
except ImportError:
    pass

try:
    from .dips import DIPSDataset
except ImportError:
    pass

try:
    from .custom_ppi import CustomPPIDataset
except ImportError:
    pass

try:
    from .confidence_dataset import ConfidenceRegressionDataset
except ImportError as e:
    import logging
    logging.warning(f'Cannot import ConfidenceRegressionDataset: {e}')

try:
    from .conformation_dataset import ConformationRegressionDataset
except ImportError as e:
    import logging
    logging.warning(f'Cannot import ConformationRegressionDataset: {e}')

try:
    from .phase3_dataset import Phase3Dataset
except ImportError as e:
    pass

try:
    from .statecontrast_structural import StateContrastStructuralDataset
except ImportError:
    pass

# get_dataset 工厂：按配置里的 type 字段查找上面已注册的数据集类
from ._base import get_dataset
