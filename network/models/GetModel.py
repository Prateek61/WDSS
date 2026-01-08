from .WDSSRegular import WDSSRegular, ModelBase
from .WDSSNoTemp import WDSSNoTemp
from .WDSSNoWavelet import WDSSNoWavelet
from .WDSSMultiHead import WDSSMultiHead
from .WDSSSWT import WDSSSWT
from .WDSSSWTSeparateBands import WDSSSWTSeparateBands
from .WDSSSWTResBlockGB import WDSSSWTResBlockGB

from typing import Dict, Any

def get_model(model_conf: Dict[str, Any]) -> ModelBase:
    if model_conf['name'] == 'WDSSRegular' and model_conf['version'] == 1.0:
        return WDSSRegular()
    if model_conf['name'] == 'WDSSNoWavelet' and model_conf['version'] == 1.0:
        return WDSSNoWavelet()
    if model_conf['name'] == 'WDSSNoTemp' and model_conf['version'] == 1.0:
        return WDSSNoTemp()
    if model_conf['name'] == 'WDSSMultiHead' and model_conf['version'] == 1.0:
        return WDSSMultiHead()
    if model_conf['name'] == 'WDSSSWT' and model_conf['version'] == 1.0:
        return WDSSSWT()
    if model_conf['name'] == 'WDSSSWTSeparateBands' and model_conf['version'] == 1.0:
        return WDSSSWTSeparateBands()
    if model_conf['name'] == 'WDSSSWTResBlockGB' and model_conf['version'] == 1.0:
        return WDSSSWTResBlockGB()
    
    raise ValueError(f"Unknown model configuration: {model_conf}")