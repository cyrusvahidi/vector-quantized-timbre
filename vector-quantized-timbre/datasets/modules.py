from typing import Callable

import gin, pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .loaders import URMP

@gin.configurable
class URMPDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        data_dir, 
        batch_size: float = 1, 
        num_workers: float = 4, 
        loader: Callable = URMP
    ):
        super().__init__()

        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader = loader

    def setup(self, stage=None):
        self.train_ds = self.loader(self.data_dir, split='train')
        self.val_ds = self.loader(self.data_dir, split='val')
        self.test_ds = self.loader(self.data_dir, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)
