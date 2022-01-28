import os
from typing import Callable


import auraloss, gin, wandb, cdpam, torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from vq_timbre.models import Encoder, Decoder, SubtractiveSynth
from vq_timbre.layers import VQEmbedding, VQEmbeddingEMA
from vq_timbre.datasets import URMPDataModule


@gin.configurable
class VQTimbreModule(pl.LightningModule):
    def __init__(
        self,
        reconstruction_loss: Callable = auraloss.freq.MultiResolutionSTFTLoss,
        loss_hyperparams = {
            'mss': 1,
            'cdpam': 0.2,
            'vq-loss': 1,
        },
        lr=0.001
    ):
        super(VQTimbreModule, self).__init__()
        
        self.loss_hyperparams = loss_hyperparams
        self.reconstruction_loss = reconstruction_loss
        self.cdpam = cdpam.CDPAM()
        self.lr = lr

        self.encoder = Encoder()
        self.vq = VQEmbeddingEMA()
        self.decoder = Decoder()
        self.synth = SubtractiveSynth()

        self.save_hyperparameters()

    def forward(self, audio, jtfs=None):
        z, g = self.encoder(audio)
        vq_z, ids, vq_losses, perplexity = self.vq(z)
        h = self.decoder(vq_z)
        y = self.synth(audio, h * g, device=audio.device)

        return y, vq_z, z, vq_losses, perplexity

    def compute_loss(self, y_pred, y_true, vq_losses):
        losses = {}
        losses['mss'] = self.reconstruction_loss(y_pred, y_true)
        losses['cdpam'] = self.cdpam.forward(y_pred, y_true.squeeze(1))
        losses['vq-loss'] = vq_losses
        total_loss = sum([v * self.loss_hyperparams[k] for k, v in losses.items()])
        losses['total'] = total_loss.mean()
        return losses

    def training_step(self, batch, batch_idx):
        item = batch
        y_true = item['audio']
        y_pred, _, _, vq_losses, perplexity = self(item['audio'], item['jtfs'])
        loss = self.compute_loss(y_pred, y_true, vq_losses)
        
        self.log_metric('train/loss', loss['total'].cpu())
        self.log_metric('train/mss', loss['mss'].cpu())
        self.log_metric('train/vq', loss['vq-loss'].cpu())
        self.log_metric('train/perplexity', perplexity)
        
        return {'loss': loss['total'], 'perplexity': perplexity}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_metric('train/loss_epoch', loss)

    def validation_step(self, batch, batch_idx):
        item = batch
        y_true = item['audio']
        y_pred, _, _, vq_losses, perplexity = self(item['audio'], item['jtfs'])
        loss = self.compute_loss(y_pred, y_true, vq_losses)

        self.log_metric('val/total', loss['total'].cpu())
        self.log_metric('val/mss', loss['mss'].cpu())
        self.log_metric('val/vq_loss', loss['vq-loss'].cpu())
        self.log_metric('val/perplexity', perplexity)
        
        return {'loss': loss['total'], 'perplexity': perplexity}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log_metric('val/loss_epoch', loss)

    def test_step(self, batch, batch_idx):
        item = batch
        y_true = item['audio']
        y_pred, _, _, vq_losses, perplexity = self(item['audio'], item['jtfs'])

        loss = self.compute_loss(y_pred, y_true, vq_losses)
        loss = loss['total']

        return {'loss': loss, 'perplexity': perplexity}

    def test_epoch_end(self, outputs):
        # keys = list(outputs[0].keys())
        # for k in keys:
        #     metric = torch.stack([x[k] for x in outputs]).mean()('test/' + k, metric)

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log_metric('test/loss_epoch', loss)

    def log_metric(self, metric_id, metric):
        self.log(metric_id, metric, prog_bar=True, on_epoch=True, logger=True)

        if self.logger:
            if hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({metric_id: metric}) # log to wandb

    def log_audio(self, audio_batch):
        for i in range(len(audio_batch)):
            audio = audio_batch[i].squeeze(0).detach().cpu().numpy()
            caption = f"Epoch {self.current_epoch} - Step {self.global_step} - Example {i}"
            self.logger.experiment.log({"examples": [wandb.Audio(audio, caption=caption, sample_rate=16000)]})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@gin.configurable
def lightning_run(
    audio_dir,
    gin_config: str = None,
    max_steps: float = 150000,
    n_epochs: float = 1,
    batch_size: float = 1,
    data_module: Callable = URMPDataModule,
    logger: bool = True
):

    dataset = data_module(audio_dir, batch_size=batch_size)

    model = VQTimbreModule()

    # Initialize a trainer
    logger = init_logger(gin_config) if logger else None
    trainer = pl.Trainer(gpus=-1,
                         max_steps=max_steps,
                         progress_bar_refresh_rate=20,
                         distributed_backend='dp',
                         logger=logger)
    device =  torch.device(f"cuda:{trainer.root_gpu}") if trainer.on_gpu and trainer.root_gpu is not None else torch.device('cpu')

    # Train the model
    trainer.fit(model, dataset)
    trainer.test(model)

def gin_register_and_parse(gin_config_file: str):
    # register MSS loss from auraloss
    gin.external_configurable(auraloss.freq.MultiResolutionSTFTLoss)

    gin.parse_config_file(gin_config_file)

def init_logger(gin_config):
    run = wandb.init()
    logger = WandbLogger(experiment=run)
    if gin_config:
        wandb.save(os.path.join(wandb.run.dir, gin_config))

    return logger
