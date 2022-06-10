r"""
Utils used by other functions and classes.
"""
import os
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union, Sequence
import pickle
from datetime import datetime


REDUCTION_TYPE = Callable[[Sequence], Any]
MODEL_TYPE = Callable[[np.ndarray], Union[np.ndarray, int]]


def load_emg_recording_dir(dir: str,
                       fillna: Union[int, float] = None) -> pd.DataFrame:
    """loads and merges emg recording from a directory into a pandas 
    DataFrame.

    Args:
        dir (str): directory containing emg recording text files
        fillna (Union[int, float], optional): value to fill missing\
            values. Defaults to None.

    Returns:
        pd.DataFrams: dataframe of the recordings. 
    """
    # search through the directory for emg recording files
    df_list = []
    for filename in os.listdir(dir):
        file = os.path.join(dir, filename)
        df_list.append(load_emg_recording(file, fillna=fillna))
    return pd.concat(df_list, ignore_index=True)


def load_emg_recording(path: str,
                       fillna: Union[int, float] = None) -> pd.DataFrame:
    """loads and merges emg recording from a text file into a pandas 
    DataFrame.

    Args:
        path (str): path to emg recording text file
        fillna (Union[int, float], optional): value to fill missing\
            values. Defaults to None.

    Returns:
        pd.DataFrams: dataframe of the recordings. 
    """
    # search through the directory for emg recording files
    df = pd.read_csv(path, sep='\t')
    # fill missing data if required
    if fillna is not None:
        df.fillna(fillna, inplace=True)
    return df


def extract_features(
        time_indexed_data: np.ndarray,
        window_size: 100,
        stats: List[REDUCTION_TYPE] = [np.min, np.max, np.mean, np.std, np.ptp]):
    n = len(time_indexed_data)
    features = []
    for i in range(n):
        start = max(0, i - window_size)
        end = min(i+ window_size, n)
        features.append(
            np.stack([fn(time_indexed_data[start:end], axis=0) for fn in stats]).flatten())     
    return np.stack(features)



def encode_model(
    model: MODEL_TYPE, window_size: str) -> Dict[str, Any]:
    return dict(model=model, window_size=window_size)


def decode_model(encoded_model: Dict[str, Any]) -> Tuple[MODEL_TYPE, int]:
    return encoded_model['model'], encoded_model['window_size']


def deploy_model(
    model: MODEL_TYPE, window_size: int, dir: str, out_name=None):
    # construct file name
    date = datetime.today().strftime("%Y-%m-%d")
    if out_name is None:
        file_name = '{}_w{}___{}'.format(
            model.__class__.__name__, window_size, date)
    else:
        file_name = out_name
    # encode model
    encoded_model = encode_model(model, window_size)
    # make sure dir exists
    if not os.path.exists(dir):
        os.makedirs(dir)
    # write
    with open(os.path.join(dir, file_name+'.pkl'), 'wb') as handle:
        pickle.dump(encoded_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(model_path: str) -> Tuple[MODEL_TYPE, int]:
    # read
    with open(model_path, 'rb') as handle:
        encoded_model = pickle.load(handle)
    return decode_model(encoded_model)


