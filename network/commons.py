
from enum import Enum

RECURSION_DEPTH = 3

@staticmethod
def wrap_try(func: callable):
    def wrapper(*args, **kwargs):

        recursion_depth = kwargs.pop('recursion_depth', RECURSION_DEPTH)

        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f'Error in {func.__name__}: {e}, depth: {recursion_depth}')

            if recursion_depth <= 0:
                raise e
            
            # Retry with decremented recursion depth
            recursion_depth -= 1
            kwargs['recursion_depth'] = recursion_depth

            return wrapper(*args, **kwargs)
        
class GB_TYPE(Enum):
    """G-Buffer types for the dataset
    """
    BASE_COLOR_DEPTH = 'BaseColorDepth'
    MV_ROUGHNESS_NOV = 'MV_Roughness_NOV'
    NORMAL_SPECULAR = 'NormalSpecular'
    PRETONEMAP_METALLIC = 'PretonemapMetallic'

class RawFrameGroup(Enum):
    HR_GB = 'HighResGBuffer'
    LR_GB = 'LowResGBuffer'
    TEMPORAL_GB = 'TemporalGBuffer'

class FrameGroup(Enum):
    """Pre processed Frame Groups for the model
    """
    GT = 'GT'
    LR_INP = 'LR_INP'
    GB_INP = 'GB_INP'
    TEMPORAL_INP = 'TEMPORAL_INP'
    EXTRA = 'EXTRA'