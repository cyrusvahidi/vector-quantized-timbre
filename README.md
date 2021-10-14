# Vector-Quantized Timbre Representation
---
Implementation of the paper [Vector-Quantized Timbre Representation](https://arxiv.org/pdf/2007.06349.pdf) - Bitton et al 2020

Developed with [Adán Benito](https://github.com/adanlbenito) early in 2021

* This repository contains a Pytorch Lightning framework to train a model similar to the one described in the paper. 
* The main difference is the use of an exponential moving-average (EMA) codebook
* With a conventional VQ codebook the codebook collapsed `perplexity = 1`.
* Data splits and preprocessing are performed for the URMP western musical instrument dataset

## Usage
```setup
pip install -r requirements.txt
pip install -e .
```

## Training

## Evaluation
