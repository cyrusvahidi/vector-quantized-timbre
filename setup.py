from setuptools import setup, find_packages

description = ('Vector-Quantized Timbre Representation Implementation')

setup(
    name='vector-quantized-timbre',
    author='Adan L. Benito, Cyrus Vahidi',
    version="0.0.1",
    packages=find_packages(include=['vq_timbre']),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchaudio',
        'torchcrepe',
        'cdpam',
        'librosa',
        'pytorch-lightning',
        'auraloss',
        'wandb',
        'gin-config',
        'resampy',
        'nnAudio'
    ]
)
