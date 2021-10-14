# Vector-Quantized Timbre Representation
---

Implementation of the paper [Vector-Quantized Timbre Representation](https://arxiv.org/pdf/2007.06349.pdf) - Bitton et al 2020

Developed with [Adán Benito](https://github.com/adanlbenito) early in 2021

* This repository contains a Pytorch Lightning framework to train a model similar to the one described in the paper. 
* The main difference is the use of an exponential moving-average (EMA) codebook
* A conventional VQ codebook caused the codebook to collapse `perplexity = 1`.

## Setup
---
```
pip install -r requirements.txt
pip install -e .
```

## Data preparation
---
Data splits and preprocessing are performed for the URMP western musical instrument dataset
To split the URMP dataset into 3 second numpy files for efficient loading:
`python scripts/urmp_numpy_segments.py <path_to_URMP> <numpy_out_dir>`

Then, set the `URMP_DATA_DIR` variable in `gin_configs/vqvae_timbre.gin` to `<numpy_out_dir>`

## Training
---
`python scripts/run_train.py`

* Most hyperparameters are configured with `gin-config`
* To log to wandb, enable `lightning_run.logger = True` in `gin_configs/vqvae_timbre.gin`

## Evaluation
--- 
* 
