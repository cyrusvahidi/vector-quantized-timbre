import os
import numpy as np
import gin
import glob
from tqdm import tqdm
import math

import pandas as pd

from typing import Union, Dict

from .tensor_utils import torch_float32, resample

def load_urmp_f0s(f0_file_path: str, sr: float):
    end_t = 0
    with open(f0_file_path, 'r') as f:
        end_t = f.readlines()[-1].split('\t')[0]
        
    f0s = np.zeros([int(float(end_t) * sr)])

    with open(f0_file_path, 'r') as f:
        last_sample = 0
        last_f0 = 0
        for line in f.readlines():
            timestamp, f0 = line.split('\t')
            sample = int(float(timestamp) * sr)

            f0s[last_sample:sample] = last_f0

            last_sample = sample
            last_f0 = float(f0)

        f0s[last_sample:-1] = last_f0

    return f0s

@gin.configurable
def get_urmp_files(data_dir: str, 
                   instr_ids: Union[str] = [],
                   audio_prefix: str = 'AuSep',
                   audio_suffix: str = '.wav',
                   f0_suffix: str = '.txt',
                   csv_path: str = None,
                   split: str = None) -> Union[Dict[str, str]]:
    ''' Gets tuples of URMP audio and corresponding f0 files
    Args:
        data_dir: the URMP data directory
        instr_ids: the URMP ids of instruments to filter or all if None
    Returns:
        files: Union[Dict[str, str]] list of dictionaries containing keys ('audio', 'f0')
    '''
    if csv_path:
        files = list_files_from_split_csv(csv_path, split, data_dir)
    else:
        files = list_files_in_path(data_dir, prefix=audio_prefix, suffix=audio_suffix)

    # filter matching audio files
    audio_files = [f for f in files 
                   if get_urmp_file_tokens(os.path.basename(f))['instr_id'] in instr_ids 
                   or not instr_ids]
    # get corresponding f0 files
    f0_files = [urmp_audio_to_f0_fname(f, audio_suffix=audio_suffix, f0_suffix=f0_suffix)
                for f in audio_files]

    # whole path
    audio_files = [os.path.join(data_dir, f) for f in audio_files]
    f0_files = [os.path.join(data_dir, f) for f in f0_files]

    # zip corresponding audio and f0 files
    audio_f0_files = zip(audio_files, f0_files)

    # convert to Union[Dict]
    audio_f0_files = [{'audio': a, 'f0': f0} 
                      for a, f0, in audio_f0_files]

    return audio_f0_files

def list_files_from_split_csv(csv_path: str, split: str = None, data_dir: str = None):
    ''' Load the files from the csv for a given split
    Args:
        csv_path: str - the absolute path to the URMP file split csv
        split: str - the data split to load, options: ['train', 'test' 'val']
        data_dir: str - the data directory to read segmented numpy files from
    '''
    df = pd.read_csv(csv_path)
    if split:
        df = df[df.split == split]

    if not data_dir:
        audio_fpaths = list(df_split.audio_fpath)
    else:
        # get the audio file names for this split
        audio_fnames = list(os.path.splitext(f)[0] for f in df.audio_fname)
        # get the segmented numpy file names from the data dir
        np_file_list = os.listdir(data_dir)
        # get all the numpy file paths for this data split
        audio_fpaths = [os.path.join(data_dir, np_f) 
                        for audio_f in audio_fnames 
                            for np_f in np_file_list 
                                if np_f.startswith(audio_f)]

    return audio_fpaths


def get_urmp_file_tokens(fname: str, sep: str = '_') -> Dict[str, str]:
    ''' Tokenize the labels contained in a URMP file name
    Returns:
        labels: Dict[str, str]
                'file_type': the URMP filetype e.g AuSep / F0s
                'stem': the identifier for the stem in a performance  
                'instr_id': the instrument identifier e.g 'vn' 'vc'
    '''
    tokens = os.path.basename(fname).split(sep)

    labels = {'file_type': tokens[0],
              'stem': tokens[1],
              'instr_id': tokens[2]}

    return labels

def urmp_audio_to_f0_fname(fpath: str,                   
                           f0_prefix: str = 'F0s',
                           audio_suffix: str = '.wav',
                           f0_suffix: str = '.txt',
                           sep: str = '_') -> str:
    ''' Converts a URMP audio file name to its corresponding f0 filename
    Returns: 
        f0_fpath: the f0 file path or name
    '''
    fname = os.path.basename(fpath)
    dirname = os.path.dirname(fpath)
    f0_fname = f0_prefix + sep + sep.join(fname.replace(audio_suffix, '').split(sep)[1:])
    f0_fname = f0_fname + f0_suffix
    f0_fpath = os.path.join(dirname, f0_fname)
    return f0_fpath

def list_files_in_path(path: str, prefix: str = 'AuSep', suffix: str = '.wav'):
    files = glob.glob(os.path.join(path, '*','{}*{}'.format(prefix, suffix)), recursive=True)
    return files

def make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Failed to create directory %s" % dir_path)

def pad_or_trim_along_axis(arr: np.ndarray, output_length: int, axis=-1):
    if arr.shape[axis] < output_length:
        n_pad = output_length - arr.shape[axis]
        
        n_dims_end = len(arr.shape) - axis - 1 if axis >= 0 else 0
        n_dims_end *= n_dims_end > 0
        
        padding = [(0, 0)] * axis + [(0, n_pad)] + [(0, 0)] * n_dims_end
        
        return np.pad(arr, padding)
    else:
        return np.take_along_axis(arr, np.arange(0, output_length, 1), axis)