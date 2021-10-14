import torch
import pytorch_lightning as pl
import fire
import gin
import wandb

from vqvae_timbre.modules import lightning_run, gin_register_and_parse


@gin.configurable
def run_train(data_dir: str = "/import/c4dm-datasets/URMP-split/npy-3-secs", 
              gin_config: str = "/homes/cv300/Documents/vq-vae-timbre/gin_configs/vqvae_timbre.gin"):
    gin_register_and_parse(gin_config)

    lightning_run(data_dir, gin_config)

def main():
  fire.Fire(run_train)

if __name__ == "__main__":
    main()
