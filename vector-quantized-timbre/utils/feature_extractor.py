import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(
        self,
        audio=None,
        fs=16000,
        wsize=2048,
        stride=512,
        center=True
    ):
        super().__init__()

        self._p = {
            'fs': fs,
            'wsize': wsize, # n_fft
            'stride': stride, # hop length
            'center': center
        }
        # Convert to mono
        if not (audio is None):
            self._p['samples'] = librosa.to_mono(audio)
            # Compute spectrogram and spectral features
            self._p['features'] = self.compute_spectral_features(self._p['samples'])

    def compute_spectrogram(self, audio):
        _complex = librosa.stft(audio,
                n_fft=self._p['wsize'],
                hop_length=self._p['stride'],
                center=self._p['center']
                )
        _mag, _phase = librosa.magphase(_complex)
        _spectrogram = {
                'complex': _complex,
                'mag': _mag,
                'phase': _phase
                }
        return _spectrogram

    def compute_spectral_features(self, audio, feat_names=None):
        _features = {}
        _audio = librosa.to_mono(audio)
        _spectrogram = self.compute_spectrogram(_audio)

        if (feat_names is None) or ('spectrogram' in feat_names):
            _features['spectrogram'] = _spectrogram
        if (feat_names is None) or ('spectral_centroid' in feat_names):
            _features['spectral_centroid'] = librosa.feature.spectral_centroid(S=_spectrogram['mag'])[0]
        if (feat_names is None) or ('spectral_bandwidth' in feat_names):
            _features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(S=_spectrogram['mag'])[0]
        if (feat_names is None) or ('spectral_contrast' in feat_names):
            _features['spectral_contrast'] = librosa.feature.spectral_contrast(S=_spectrogram['mag'], fmin=200.0, n_bands=6, quantile=0.02)
        if (feat_names is None) or ('spectral_flatness' in feat_names):
            _features['spectral_flatness'] = librosa.feature.spectral_flatness(S=_spectrogram['mag'], amin=1e-10)[0]
        if (feat_names is None) or ('spectral_rolloff' in feat_names):
            _features['spectral_rolloff'] = librosa.feature.spectral_rolloff(S=_spectrogram['mag'], roll_percent=0.85)[0]
        if (feat_names is None) or ('f0' in feat_names):
            _features['fundamental_f'] = self.extract_f0(_audio, fs=self._p['fs'])
            _features['f0'] = _features['fundamental_f']['f0']

        return _features

    def extract_f0(self, audio, fs, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), method='pyin'):
        _fundamental = {}
        if method == 'pyin':
            _f0, _voiced_flags, _voiced_probs = librosa.pyin(
                    audio,
                    fmin=fmin,
                    fmax=fmax,
                    sr=fs,
                    frame_length=self._p['wsize'],
                    hop_length=self._p['stride']
                    )
            _fundamental['method'] = method
            _fundamental['f0'] = _f0
            _fundamental['voiced_flags'] = _voiced_flags
            _fundamental['voiced_probs'] = _voiced_probs
        return _fundamental

    def plot_spectrogram(self, features=None, ax=None, title='Log Power Spectrogram'):
        if not features:
            features = self._p['features']

        if not ax:
            fig, ax = plt.subplots()

        img = librosa.display.specshow(
            librosa.amplitude_to_db(
                features['spectrogram']['mag'], ref=np.max),
                y_axis='log',
                x_axis='time',
                ax=ax
            )
        if title:
            ax.set(title=title)
        return img

    def plot_feature(self, feature_name, features=None, ax=None, fig=None, do_return=False):
        if not features:
            features = self._p['features']

        if feature_name == 'spectral_centroid':
            if not ax:
                fig, ax = plt.subplots()
            _centroid = features['spectral_centroid']
            times = librosa.times_like(_centroid)
            self.plot_spectrogram(features=features, ax=ax)
            ax.plot(times, _centroid, label='Spectral centroid', color='w')
            ax.legend(loc='upper right')

            if do_return:
                return _centroid, times

        elif feature_name == 'spectral_bandwidth':
            if not ax:
                fig, ax = plt.subplots(nrows=2, sharex=True)
            _bw = features['spectral_bandwidth']
            times = librosa.times_like(_bw)
            if len(ax) >= 2:
                ax[0].semilogy(times, _bw, label='Spectral bandwidth')
                ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
                ax[0].legend()
                ax[0].label_outer()
                _ax = ax[1]
            else:
                _ax = ax
            self.plot_spectrogram(features=features, ax=_ax)
            _centroid, _ = self.plot_feature('spectral_centroid', features=features, ax=_ax, do_return=True)
            _ax.fill_between(times, _centroid - _bw, _centroid + _bw, alpha=0.5, label='Centroid +- bandwidth')
            _ax.legend(loc='lower right')

            if do_return:
                return _bw, times

        elif feature_name == 'spectral_contrast':
            if not ax:
                fig, ax = plt.subplots(nrows=2, sharex=True)
            _contrast = features['spectral_contrast']
            img1 = self.plot_spectrogram(features=features, ax=ax[0])
            if fig:
                fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
            img2 = librosa.display.specshow(_contrast, x_axis='time', ax=ax[1])
            if fig:
                fig.colorbar(img2, ax=[ax[1]])
            ax[1].set(ylabel='Frequency bands', title='Spectral contrast')

            if do_return:
                return _contrast

        elif feature_name == 'spectral_rolloff':
            if not ax:
                fig, ax = plt.subplots()
            _rolloff = features['spectral_rolloff']
            times = librosa.times_like(_rolloff)
            self.plot_spectrogram(features=features, ax=ax)
            ax.plot(times, _rolloff, label='Roll-off frequency (0.85)')
            ax.legend(loc='lower right')

            if do_return:
                return _rolloff, times

        elif feature_name == 'f0':
            if not ax:
                fig, ax = plt.subplots()
            _f0 = features['f0']
            times = librosa.times_like(_f0)
            img = self.plot_spectrogram(features=features, ax=ax)
            ax.plot(times, _f0, label='f0', color='cyan', linewidth=3)
            ax.legend(loc='upper right')

            if do_return:
                return _f0, times
        else:
            return -1

    def is_array(self, var):
        _arrayFlag = False
        _var = var
        if isinstance(_var, (list,tuple,np.ndarray)):
            _var = np.asarray(_var)
            _arrayFlag = True
        return _arrayFlag, _var


    def get_basic_stats(self, features, axis=-1):
        _stats = {}
        _axis = axis
        _stats_dict = {}
        for ft_name, ft_val in features.items():
            _isarray, value = self.is_array(ft_val)
            if _isarray:
                _stats_dict[ft_name] = {
                    'mean': np.nanmean(value, axis=_axis),
                    'std': np.nanstd(value, axis=_axis)
                    }
        return _stats_dict

    def get_features(self, feat_names = None):
        _feats = self._p['features']
        if feat_names:
            _feats = dict(filter(lambda e: e[0] in feat_names, _feats.items()))
        return _feats
