import numpy as np
from vqvae_timbre.utils import FeatureExtractor
from vqvae_timbre.models import Encoder, Decoder, Subtractive
import torch

class FeatureMap:
    def __init__(
        trained_model,
        n_frames=10,
        feat_names=['f0', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 'spectral_rolloff']
    ):
        self._p = {
            'n_frames': n_frames,
            'feat_names': feat_names
        }

        self.embedding = trained_model.vq.embedding
        self.encoder = trained_model.encoder
        self.decoder = trained_model.decoder
        self.synth = trained_model.synth

        self._p['n_codes'] = self.embedding.weight.shape[0]
        self._p['gain'] = torch.ones(self._p['n_codes'], self._p['n_frames'], 1)
        self._p['stride'] = self.synth._p['stride']
        self._p['wsize'] = self.synth._p['wsize']
        self._p['fs'] = self.synth._p['fs']

        self.feat_extract = FeatureExtractor(fs=self._p['fs'])


    def get_features(self):
        vq_samples = torch.stack([self.embedding.weight for i in range(self._p['n_frames'])]).permute(1, 0, 2)
        coeffs = self.decoder(vq_samples)
        gen_audio = self.synth(torch.zeros(self._p['n_codes'], 1, (self._p['n_frames'] - 1) * self._p['stride']), self._p['gain'] * coeffs)

        gen_audio_np = gen_audio.detach().numpy().astype(np.float32)
        features = [self.feat_extract.compute_spectral_features(x, feat_names=self._p['feat_names']) for x gen_audio_np]

        return features, vq_samples

    def mapping(self):
        features, vq_samples = self.get_features()
        #feat_stats = self.feat_extract.get_basic_stats(features)

        return None



