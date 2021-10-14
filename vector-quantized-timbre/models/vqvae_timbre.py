''' Implementation/interpretation of Bitton's VQ-VAE '''

from typing import Union

import numpy as np, gin, torch, torchaudio, torch.nn as nn, torch.nn.functional as nnF
from nnAudio import Spectrogram

from vqvae_timbre.utils import resample


class Encoder(nn.Module):

    def __init__(
        self,
        latent_features=128,
        codebook_size=1024,
        wsize=2048,
        stride=512,
        downsampling_out_dim=[32, 64, 96, 128, 192, 224, 256],
        downsampling_stride=2,
        downsampling_kernel_size=13,
        scalar_gain=False,
        device='cuda'
    ):
        super().__init__()
        self._p = {
            'latent_features': latent_features, # d_z
            'codebook_size': codebook_size, # K
            'wsize': wsize, # L
            'stride': stride, # S
            'downsampling_out_dim': downsampling_out_dim,
            'downsampling_stride': downsampling_stride,
            'downsampling_kernel_size': downsampling_kernel_size,
            'n_coefficients': (wsize + 2) // 2, # N
            'scalar_gain': scalar_gain
        }
        self.device = device
        # Slice input in overlapping windows using 1D convolution with hann kernel
        self.hann_w = torch.hann_window(self._p['wsize'])
        self.hann_w = torch.reshape(self.hann_w, (1, 1, self._p['wsize']))

        self.slicer = nn.Unfold(
            kernel_size=(1, self._p['wsize']),
            stride=self._p['stride'],
            padding=(0, wsize // 2)
        )

        # Convolutional downsampling block
        self.downsampling_conv = nn.ModuleList([])
        last_out_dim = 1
        for dim in self._p['downsampling_out_dim']:
            self.downsampling_conv.append(
                nn.Conv1d(
                    in_channels=last_out_dim,
                    out_channels=dim,
                    kernel_size=self._p['downsampling_kernel_size'],
                    stride=self._p['downsampling_stride']
                )
            )
            last_out_dim = dim

        # Output linear layers
        self.out_layers = nn.ModuleList([])

        self._p['bottleneck_dims'] = self.get_bottleneck_dims()[0]
        # Latent features
        self.latent_output = nn.Linear(in_features=self._p['bottleneck_dims'], out_features=self._p['latent_features'])
        # Gains
        if self._p['scalar_gain']:
            self.gain_output = nn.Linear(in_features=self._p['bottleneck_dims'], out_features=1)
        else:
            self.gain_output = nn.Linear(in_features=self._p['bottleneck_dims'], out_features=self._p['n_coefficients'])

    def get_bottleneck_dims(self):
        # Dummy Window
        dummy = torch.empty((1, 1, self._p['wsize']))
        for i in range(len(self.downsampling_conv)):
            dummy = self.downsampling_conv[i](dummy)
        dummy  = torch.flatten(dummy)
        return dummy.shape

    def get_params(self):
        return self._p

    def splice_tensor(self, samples):
        samples = samples.unsqueeze(1)
        slices = self.slicer(samples)
        slices = slices.permute([0, -1, 1])
        return slices

    def forward(self, samples, device=None):
        x = self.splice_tensor(samples) * self.hann_w.to(device)
        x = torch.reshape(x, (-1, 1, x.shape[-1]))
        for i in range(len(self.downsampling_conv)):
            x = self.downsampling_conv[i](x)
        x = torch.flatten(x, 1)
        z = self.latent_output(x)
        g = self.gain_output(x)

        z = z.reshape((samples.shape[0], -1, z.shape[-1]))
        g = g.reshape((samples.shape[0], -1, g.shape[-1]))

        return z, g


class Decoder(nn.Module):

    def __init__(
        self, 
        latent_features=128, 
        codebook_size=1024,  
        wsize=2048, 
        n_hidden_l=4, 
        n_gru_l=1, 
        hidden_dim=768
    ):
        super().__init__()
        self._p = {
            'latent_features': latent_features, # d_z
            'codebook_size': codebook_size, # K
            'wsize': wsize, # L
            'n_coefficients': (wsize + 2) // 2, # N
            'n_hidden_l': n_hidden_l,
            'n_gru_l': n_gru_l,
            'hidden_dim': hidden_dim
        }
        # Input Stack of Linear Layers
        self.linear_stack_1 = nn.ModuleList([nn.Linear(self._p['latent_features'], self._p['hidden_dim'])])
        for i in range(self._p['n_hidden_l']-1):
            self.linear_stack_1.append(nn.Linear(self._p['hidden_dim'], self._p['hidden_dim']))

        # RNN
        self.rnn_net = nn.GRU(self._p['hidden_dim'], self._p['hidden_dim'], num_layers=self._p['n_gru_l'], batch_first=True)

        # Output Stack of Linear Layers
        self.linear_stack_2 = nn.ModuleList([nn.Linear(self._p['hidden_dim'], self._p['hidden_dim']) for i in range(self._p['n_hidden_l'])])
        self.output_layer = nn.Linear(self._p['hidden_dim'], self._p['n_coefficients'])
        self.sigmoid = nn.Sigmoid()

    def get_params(self):
        return self._p

    def forward(self, samples, verbose=False):
        x = samples

        for i in range(len(self.linear_stack_1)):
            x = self.linear_stack_1[i](x)
        if verbose:
            print("Linear stack [1] output shape: ", x.shape)

        x, h_state = self.rnn_net(x)
        if verbose:
            print("GRU output shape: ", x.shape)

        for i in range(len(self.linear_stack_2)):
            x = self.linear_stack_2[i](x)
        if verbose:
            print("Linear stack [2] output shape: ", x.shape)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        x = torch.log1p(x)
        if verbose:
            print("Output layer shape: ", x.shape)
        return x


class SubtractiveSynth(nn.Module):

    def __init__(
        self, 
        fs=16000, 
        wsize=2048, 
        stride=512, 
        trainable=False, 
        device=None
    ):
        super().__init__()
        self._p = {
            'fs': fs,
            'wsize': wsize, # L
            'stride': stride, # S
            'n_coefficients': (wsize//2)+1, # N
            'trainable': trainable
        }
        '''
        STFT layer:
            n_fft: L,
            freq_bins: L+2,
            hop_length: S,
            output_dim: Complex -> [num_samples, freq_bins, time_steps, 2]
        '''
        self.stft_layer = Spectrogram.STFT(
            n_fft=self._p['wsize'],
            freq_bins=None,
            hop_length=self._p['stride'],
            sr=self._p['fs'],
            trainable=self._p['trainable'],
            output_format='Complex',
            center=True,
            iSTFT=True
        )

    def forward(self, samples, coefficients=None, device=None):
        '''
        Args:
            samples: [batch, 1, n_audio_samples]
            magnitudes: frequency envelope to generate an impulse response from [batch, n_frequencies, n_frames]
        '''
        noise = self.gen_noise(samples.shape).to(device)
        x = self.stft_layer(noise)
        stft_complex = torch.complex(x[:,:,:,0], x[:,:,:,1])
        if coefficients is not None:
            coefficients = resample(coefficients.permute(0,2,1), stft_complex.shape[-1])
            coefficients_complex = torch.complex(coefficients, torch.zeros_like(coefficients))
            coefficients_complex = coefficients_complex
            x = stft_complex * coefficients_complex
            x = torch.stack([x.real, x.imag], dim=-1)
        audio = self.stft_layer.inverse(x, length=samples.shape[-1])
        return audio

    def gen_noise(self, dim, type='uniform', device=None):
        if type == 'uniform':
            self.noise = torch.rand(dim) * 2 - 1
        return self.noise

    def get_params(self):
        return self._p