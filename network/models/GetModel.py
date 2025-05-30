from .WDSSRegular import WDSSRegular, ModelBase
from .WDSSNoTemp import WDSSNoTemp

from typing import Dict, Any

def get_model(model_conf: Dict[str, Any]) -> ModelBase:
    if model_conf['name'] == 'WDSSRegular' and model_conf['version'] == 1.0:
        return WDSSRegular()
    if model_conf['name'] == 'WDSSNoTemp' and model_conf['version'] == 1.0:
        return WDSSNoTemp()
    
    raise ValueError(f"Unknown model configuration: {model_conf}")