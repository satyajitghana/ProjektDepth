from .base_trainer import BaseTrainer
import vathos.data_loader as data_loader
from vathos.utils import setup_logger

import gc
import torch
from tqdm.auto import tqdm
import vathos.model.loss as vloss


logger = setup_logger(__name__)


class GPUTrainer(BaseTrainer):
    r'''
    GPUTrainer: Trains the vathos model on GPU
    '''

    def __init__(self, *args, **kwargs):
        super(GPUTrainer, self).__init__(*args, **kwargs)

        # set the device to GPU:0 // we don't support multiple GPUs for now
        self.device = torch.device("cuda:0")

    def train_epoch(self, epoch):

        # clear the cache before training this epoch
        gc.collect()
        torch.cuda.empty_cache()

        pbar = tqdm(self.train_loader, dynamic_cols=True)

        # set the model to training mode
        self.model.train()

        for batch_idx, data in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            data = getattr(data_loader, self.config['data_loader']['type']).apply_on_batch(
                data,
                lambda x: x.to(self.device)
            )

            # zero out the gradients, we don't want to accumulate them
            self.optimizer.zero_grad()

            x = torch.cat([data['bg'], data['fg_bg']], dim=1)
            d_out, s_out = self.model(x)

            # calculate the losses
            l1 = self.seg_loss(s_out, data['fg_bg_mask'])
            l2 = self.depth_loss(d_out, data['depth_fg_bg'])

            loss = self.comb_loss(l1, l2)

            # update the gradients
            loss.backward()

            # step the optmizer
            self.optimizer.step()

            pbar.set_description(
                desc=f'loss={loss.item():.4f} seg_loss={l1.item():.4f} depth_loss={l2.item():.4f} batch_id={batch_idx}')

    def test_epoch(self, epoch):

        # clear the cache before testing this epoch
        gc.collect()
        torch.cuda.empty_cache()

        # set the model in eval mode
        self.model.eval()

        miou = 0
        mrmse = 0

        pbar = tqdm(self.test_loader, dynamic_cols=True)

        for batch_idx, data in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            data = getattr(data_loader, self.config['dataset']).apply_on_batch(
                data,
                lambda x: x.to(self.device)
            )

            x = torch.cat([data['bg'], data['fg_bg']], dim=1)

            with torch.no_grad():
                d_out, s_out = self.model(x)
                miou += vloss.iou(s_out, data['fg_bg_mask'])
                mrmse += vloss.rmse(d_out, data['depth_fg_bg'])

            pbar.set_description(desc=f'testing batch_id={batch_idx}')

        miou /= len(pbar)
        mrmse /= len(pbar)
