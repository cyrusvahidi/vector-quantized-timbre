import torch
import torch.nn.functional as F
import gin

from typing import Union, Callable

@gin.register
def resample(signal, n_samples):
    is_1d = len(signal.shape) == 1
    is_2d = len(signal.shape) == 2
    if (is_1d):
        signal = signal.view(1, 1, -1)
    if is_2d:
        signal = signal.unsqueeze(0)
    
    signal = torch.nn.functional.interpolate(signal, size=n_samples, mode='linear', align_corners=False)
    
    if is_1d:
        signal = signal[0, 0, :]
    elif is_2d:
        signal = signal[0, :, :]
    
    return signal


@gin.register
def torch_float32(x):
    '''
        Convert a numpy array or torch tensor 
        to a 32-bit torch float tensor
    '''
    if isinstance(x, torch.FloatTensor):
        return x
    elif isinstance(x, torch.Tensor):
        return x.type(torch.FloatTensor)
    else:
        return torch.from_numpy(x).type(torch.FloatTensor)


@gin.configurable
def prepare_input_tensor(x: torch.Tensor, preprocessors: Union[Callable]):
    ''' Prepare data tensors for input to the network 
        with a series of preprocessing functions
        1. Add the channel dimension
        2. convert to float32 tensor
    '''
    for processor in preprocessors:
        x = processor(x)
    return x

@gin.register
def add_channel_dim(x: torch.Tensor):
    ''' Adds a channel dimension to unbatched data
    '''
    return x.unsqueeze(0)

def pad_axis(x, padding=(0, 0), axis=1):
    ''' ddsp/core.py
        pad
    '''
    n_end_dims = len(x.shape) - axis - 1
    n_end_dims *= n_end_dims > 0
    paddings = [[0, 0]] * axis + [list(padding)] + [[0, 0]] * n_end_dims
    paddings = tuple(elem for tupl in paddings for elem in tupl)
    # paddings = [tuple(padding) if i == axis else (0, 0) for i in range(len(x.shape))]
    return F.pad(x, paddings)