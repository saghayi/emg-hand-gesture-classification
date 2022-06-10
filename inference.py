r"""
This files contains inference methods used in post deployment stage.
"""

import numpy as np
from typing import Callable, Sequence, Union
from utils import extract_features


MODEL_TYPE = Callable[[np.ndarray], Union[np.ndarray, int]]

def inference(
    buffer: Sequence[np.ndarray], 
    model: MODEL_TYPE, window: int) -> Sequence[int]:
    extracted_features = extract_features(np.stack(buffer), window)
    return model.predict(
        extracted_features)[min(0, len(extracted_features)-window)
        ]

