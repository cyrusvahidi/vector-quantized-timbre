from vqvae_timbre.utils import FeatureExtractor
from vqvae_timbre.models import Encoder, Decoder, SubtractiveSynth
import torch
import numpy as np
import pandas as pd
from collections import namedtuple

FeatMapping = namedtuple('FeatMapping',['feature_name','values','codes'])

class FeatureMap:
    def __init__(self,
        trained_model,
        n_frames =  10,
        feat_names = ['f0', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness', 'spectral_rolloff'],
        verbose=True
    ):
        self._p = {
            'n_frames': n_frames,
            'feat_names': feat_names
        }

        self.verbose = verbose

        self.model = trained_model

        self.embedding = self.model.vq.embedding
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.synth = self.model.synth

        self._p['n_codes'] = self.embedding.weight.shape[0]
        self._p['gain'] = torch.ones(self._p['n_codes'], self._p['n_frames'], 1)
        self._p['stride'] = self.synth._p['stride']
        self._p['wsize'] = self.synth._p['wsize']
        self._p['fs'] = self.synth._p['fs']

        self.feat_extract = FeatureExtractor(fs=self._p['fs'])

        self.map_dict = None

    def get_features(self):
        if self.verbose: print('Computing filter predictions for %d frames ...\n' % self._p['n_frames'])
        vq_samples = torch.stack([self.embedding.weight for i in range(self._p['n_frames'])]).permute(1, 0, 2)
        coeffs = self.decoder(vq_samples)
        if self.verbose: print('Generating audio samples ...\n')
        gen_audio = self.synth(torch.zeros(self._p['n_codes'], 1, (self._p['n_frames'] - 1) * self._p['stride']), self._p['gain'] * coeffs)
        gen_audio_np = gen_audio.detach().numpy().astype(np.float32)
        if self.verbose: print('Extracting features ...\n')
        features = [self.feat_extract.compute_spectral_features(x, feat_names=self._p['feat_names']) for x in gen_audio_np]

        return features, vq_samples

    '''
    Possible output formats: 'dict', 'tuple' (list of named dictionaries), 'dataframe'
    '''
    def mapping(self, features, vq_samples, output_format='dict'):
        if self.verbose: print("Extracting mappings...\n")
        ft_maps = []
        for idx, code in enumerate(vq_samples):
            stats = self.feat_extract.get_basic_stats(features[idx])
            for ft, st in stats.items():
                ft_map = FeatMapping(feature_name=ft, values=st['mean'], codes=code[0])
                ft_maps.append(ft_map)
        map_df = pd.DataFrame(ft_maps)
        del ft_maps   
        # Dictionary
        feat_map_dict = dict.fromkeys(map_df.feature_name.unique())
        if output_format == 'tuple': 
            # Named tuple array
            feat_map_tuples = []
        for ft in map_df.feature_name.unique():
            query_string = 'feature_name == "' + str(ft) + '"'
            ft_subset = map_df.query(query_string)[['values', 'codes']]
            ft_dict = ft_subset.to_dict('list')
            feat_map_dict[ft] = ft_dict
            if output_format == 'tuple':
                ft_tuple= FeatMapping(feature_name=ft, values=ft_dict['values'], codes=ft_dict['codes'])
                feat_map_tuples.append(ft_tuple)
       
        if output_format == 'dataframe':
            return pd.DataFrame(feat_map_dict).T 
        elif output_format == 'tuple':
            return feat_map_tuples
        elif output_format == 'dict':
            return feat_map_dict
        else:
            return None


    def get_feature_mapping(self, rerrun=False):
        if not (hasattr(self, 'map_dict')) or (self.map_dict is None) or rerrun:
            if self.verbose: print("Generating new mapping for this model.\n")
            features, vq_samples = self.get_features()
            map_dict = self.mapping(features, vq_samples, output_format='dict')
            self.map_dict =  map_dict
        else:
            if self.verbose: print("Mapping already exists for this model.\n")
        return self.map_dict


    def get_target_code(self, feature_name, target_value):
        if not (hasattr(self, 'map_dict')) or (self.map_dict is None):
            raise ValueError('Mapping needs to be performed before calling this method using FeatureMap.get_feature_mapping()')
        if not (feature_name in self.map_dict):
            raise ValueError('The specified feature has not been computed by the feature extractor')
        
        _ft = np.array(self.map_dict[feature_name]['values'])
        codes = self.map_dict[feature_name]['codes']
        ft_dist = np.abs(_ft - target_value)
        code_id = np.nanargmin(ft_dist)
        return codes[code_id], _ft[code_id]

    def get_feature_range(self, feature_name):
        if not (hasattr(self, 'map_dict')) or (self.map_dict is None):
            raise ValueError('Mapping needs to be performed before calling this method using FeatureMap.get_feature_mapping()')
        if not (feature_name in self.map_dict):
            raise ValueError('The specified feature has not been computed by the feature extractor')
        values = self.map_dict[feature_name]['values']
        return np.nanmin(values), np.nanmax(values)

    def get_avg_code(self, feature_names, target_values): 
        if len(feature_names) != len(target_values):
            raise ValueError('Number of given values must match number of features')
        codes = list(map(lambda i:get_target_code(feature_names[i], target_values[i])[0], range(0, len(feature_names))))   
        mean_code = torch.mean(torch.stack(codes), dim=0)
        return mean_code

    def get_code_series(self, feature_name, target_list):
        if not (hasattr(self, 'map_dict')) or (self.map_dict is None):
            raise ValueError('Mapping needs to be performed before calling this method using FeatureMap.get_feature_mapping()')
        if not (feature_name in self.map_dict):
            raise ValueError('The specified feature has not been computed by the feature extractor')

        descriptor_vals = list(map(lambda x: self.get_target_code(feature_name, x)[1], target_list))
        codes = list(map(lambda x: self.get_target_code(feature_name, x)[0], target_list))

        return codes, descriptor_vals
