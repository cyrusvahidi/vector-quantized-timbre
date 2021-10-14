import os
from typing import Union, Callable

import librosa, numpy as np, torch, torchaudio, gin
from torch.utils.data import Dataset, DataLoader

from vqvae_timbre.utils import (
    get_urmp_files, 
    get_urmp_file_tokens,
    resample, 
    prepare_input_tensor,
    list_files_in_path, 
    pad_or_trim_along_axis,
    load_numpy, 
    list_files_from_split_csv, 
    torch_float32
)

@gin.configurable
class URMP(Dataset):

    def __init__(
        self, 
        data_dir: str, 
        sr: float = 16000, 
        input_length: float = 3,
        instr_ids: Union[str] = ['vn'],
        audio_load_fn: Callable = load_numpy,
        csv_path: str = None,
        split: str = None
    ):
        super(URMP, self).__init__()

        self.data_dir = data_dir
        self.sr = sr 
        self.audio_load_fn = audio_load_fn

        csv_path = os.path.join(os.getcwd(), csv_path)

        self.urmp_files = get_urmp_files(data_dir, instr_ids, csv_path=csv_path, split=split)

        self.n_items = len(self.urmp_files)
    
        self.input_length = input_length * self.sr if input_length else None
        self.instr_ids = instr_ids
        self.csv_path = csv_path 
        self.split = split

    def __getitem__(self, idx):
        f_path = self.urmp_files[idx]['audio']

        # load and prepare audio file        
        audio = self.audio_load_fn(f_path)
        audio = prepare_input_tensor(audio)

        return audio
        
    def __len__(self):
        return self.n_items