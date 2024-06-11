import os
from opt import get_opts
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.profiler import PyTorchProfiler

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk # chunk size is effective in val mode
                # self.train_dataset.dataset.white_back
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.full_train_dataset = self.train_dataset
        self.val_dataset = dataset(split='val', **kwargs)

        # # Sample a subset of the training dataset
        # num_samples = 20480  # Number of samples to use
        # train_indices = np.random.choice(len(self.train_dataset), num_samples, replace=False)
        # self.train_dataset = Subset(self.train_dataset, train_indices)

        # # Manually add attributes to the subset
        # self.train_dataset.dataset.white_back = self.train_dataset.dataset.white_back

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        # Resample the training dataset at the beginning of each epoch
        num_samples = 20480  # Number of samples to use
        train_indices = np.random.choice(len(self.full_train_dataset), num_samples, replace=False)
        self.train_dataset = Subset(self.full_train_dataset, train_indices)

        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def on_train_epoch_start(self):
        # Resample the training dataset at the beginning of each epoch
        num_samples = 20480  # Number of samples to use
        train_indices = np.random.choice(len(self.full_train_dataset), num_samples, replace=False)
        self.train_dataset = Subset(self.full_train_dataset, train_indices)

        # # Manually add attributes to the subset
        # self.train_dataset.dataset.white_back = self.full_train_dataset.white_back

        # Update the dataloader
        self.train_dataloader_object = DataLoader(self.train_dataset,
                              shuffle=True,
                              num_workers=4,
                              batch_size=self.hparams.batch_size,
                              pin_memory=True)

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        self.log('train/loss', loss)
        self.log('train/psnr', psnr_)

        return {'loss': loss, 'progress_bar': {'train_psnr': psnr_}, 'log': log}

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_value = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('val/psnr', psnr_value, on_epoch=True, prog_bar=True)
        
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        return {'val_loss': loss, 'val_psnr': psnr_value}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss, on_epoch=True, prog_bar=True)
        self.log('val/psnr', mean_psnr, on_epoch=True, prog_bar=True)

        return {
            'progress_bar': {'val/loss': mean_loss, 'val/psnr': mean_psnr},
            'log': {'val/loss': mean_loss, 'val/psnr': mean_psnr}
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, monitor_val=None):
        scheduler.step()

if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath='/content/drive/MyDrive/nerf_checkpoints',  # Google Drive path
        filename='{epoch:02d}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
    )

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        weights_summary=None,
        progress_bar_refresh_rate=1,
        gpus=hparams.num_gpus,
        strategy='ddp' if hparams.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler=PyTorchProfiler()
    )

    trainer.fit(system)
