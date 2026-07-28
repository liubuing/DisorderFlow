# Eager imports — @register_dataset decorators run at import time
# Wrap heavy-dependency imports with error handling

from .protein import preprocess_protein_structure

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

from ._base import get_dataset
