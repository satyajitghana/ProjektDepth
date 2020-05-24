from .base_trainer import BaseTrainer, optimizer_to, scheduler_to
import vathos.data_loader as vdata_loader
from vathos.utils import setup_logger
import vathos.model.loss as vloss

import gc
import torch
from pathlib import Path
# from tqdm.auto import tqdm
from tqdm.notebook import tqdm, trange
import torch.optim as optim
import torch.utils as utils

logger = setup_logger(__name__)


class GPUTrainer(BaseTrainer):
    r'''
    GPUTrainer: Trains the vathos model on GPU
    '''

    def __init__(self, *args, **kwargs):
        super(GPUTrainer, self).__init__(*args, **kwargs)

        cfg = self.config

        # set the device to GPU:0 // we don't support multiple GPUs for now
        self.device = torch.device("cuda:0")

        self.writer.add_graph(self.model, (torch.randn(1, 6, 96, 96)))
        self.writer.flush()

        self.model = self.model.to(self.device)

        optimizer_to(self.optimizer, self.device)
        scheduler_to(self.lr_scheduler, self.device)

    def train_epoch(self, epoch):

        logger.info(f'=> Training Epoch {epoch}')

        # clear the cache before training this epoch
        gc.collect()
        torch.cuda.empty_cache()

        # pbar = tqdm(self.train_loader, dynamic_ncols=True)
        pbar = self.train_loader

        # set the model to training mode
        self.model.train()

        miou = 0
        mrmse = 0
        seg_loss = 0
        depth_loss = 0

        for batch_idx, data in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            data = getattr(vdata_loader, self.config['dataset']['name']).apply_on_batch(
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

            with torch.no_grad():
                miou += vloss.iou(s_out, data['fg_bg_mask'])
                mrmse += vloss.rmse(d_out, data['depth_fg_bg'])

            # update the gradients
            loss.backward()

            # step the optmizer
            self.optimizer.step()

            # step the scheduler
            if isinstance(self.lr_scheduler, optim.lr_scheduler.OneCycleLR):
                self.lr_scheduler.step()

            seg_loss += l1.item()
            depth_loss += l2.item()

            # pbar.set_description(
            #     desc=f'loss={loss.item():.4f} seg_loss={l1.item():.4f} depth_loss={l2.item():.4f} batch_id={batch_idx}')

            self.writer.add_scalar(
                'BatchLoss/Train/seg_loss', l1.item(), epoch*len(pbar) + batch_idx)
            self.writer.add_scalar(
                'BatchLoss/Train/depth_loss', l2.item(), epoch*len(pbar) + batch_idx)

        seg_loss /= len(pbar)
        depth_loss /= len(pbar)
        miou /= len(pbar)
        mrmse /= len(pbar)

        logger.info(
            f'seg_loss: {seg_loss}, depth_loss: {depth_loss}, mIOU: {miou}, mRMSE: {mrmse}')

        self.writer.flush()

        return {'miou': miou, 'mrmse': mrmse, 'seg_loss': seg_loss, 'depth_loss': depth_loss}

    def test_epoch(self, epoch):
        logger.info(f'=> Testing Epoch {epoch}')

        # clear the cache before testing this epoch
        gc.collect()
        torch.cuda.empty_cache()

        # set the model in eval mode
        self.model.eval()

        miou = 0
        mrmse = 0
        seg_loss = 0
        depth_loss = 0

        # pbar = tqdm(self.test_loader, dynamic_ncols=True)
        pbar = self.test_loader

        for batch_idx, data in enumerate(pbar):
            # move the data of the specific dataset to our `device`
            data = getattr(vdata_loader, self.config['dataset']['name']).apply_on_batch(
                data,
                lambda x: x.to(self.device)
            )

            x = torch.cat([data['bg'], data['fg_bg']], dim=1)

            with torch.no_grad():
                d_out, s_out = self.model(x)
                miou += vloss.iou(s_out, data['fg_bg_mask'])
                mrmse += vloss.rmse(d_out, data['depth_fg_bg'])

                l1 = self.seg_loss(s_out, data['fg_bg_mask'])
                l2 = self.depth_loss(d_out, data['depth_fg_bg'])

                seg_loss += l1.item()
                depth_loss += l2.item()

            # pbar.set_description(desc=f'testing batch_id={batch_idx}')

        miou /= len(pbar)
        mrmse /= len(pbar)
        seg_loss /= len(pbar)
        depth_loss /= len(pbar)

        logger.info(f'mIOU: {miou} mRMSE: {mrmse}')

        results = {**data, 'pred_depth': d_out, 'pred_mask': s_out}

        return {'miou': miou, 'mrmse': mrmse, 'seg_loss': seg_loss, 'depth_loss': depth_loss, 'results': results}

    def start_train(self):
        logger.info('=> Training Started')
        logger.info(f'Training the model for {self.epochs} epochs')

        for epoch in range(self.start_epoch, self.epochs):
            if self.lr_scheduler:
                lr_value = [group['lr']
                            for group in self.optimizer.param_groups][0]

                logger.info(f'=> LR was set to {lr_value}')
                self.writer.add_scalar('LR/lr_value', lr_value, epoch)

            # train this epoch
            train_metric = self.train_epoch(epoch)

            self.writer.add_scalar(
                'EpochLoss/Train/seg_loss', train_metric['seg_loss'], epoch)
            self.writer.add_scalar(
                'EpochLoss/Train/depth_loss', train_metric['depth_loss'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Train/mIOU', train_metric['miou'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Train/mRMSE', train_metric['mrmse'], epoch)

            # test this epoch
            test_metric = self.test_epoch(epoch)

            self.writer.add_scalar(
                'EpochLoss/Test/seg_loss', test_metric['seg_loss'], epoch)
            self.writer.add_scalar(
                'EpochLoss/Test/depth_loss', test_metric['depth_loss'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Test/mIOU', test_metric['miou'], epoch)
            self.writer.add_scalar(
                'EpochAccuracy/Test/mRMSE', test_metric['mrmse'], epoch)

            test_images = getattr(vdata_loader, self.config['dataset']['name']).plot_results(
                test_metric['results'])

            self.writer.add_figure(
                'ModelImages/TestImages', test_images, epoch)

            # check if we improved accuracy and save the model
            if (test_metric['mrmse'] <= self.best_accuracy['mrmse']) or (test_metric['miou'] >= self.best_accuracy['miou']):

                self.best_accuracy['mrmse'] = test_metric['mrmse']
                self.best_accuracy['miou'] = test_metric['miou']

                logger.info('=> Accuracy improved, saving checkpoint ...')

                chkpt_path = Path(self.config['chkpt_dir'])
                chkpt_path.mkdir(parents=True, exist_ok=True)

                model_checkpoint = chkpt_path / 'model_checkpoint.pt'
                train_checkpoint = chkpt_path / 'train_checkpoint.pt'
                torch.save(self.model.state_dict(), model_checkpoint)

                torch.save({
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.lr_scheduler.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'save_epoch': epoch,
                    'total_epochs': self.epochs
                }, train_checkpoint)

            self.writer.flush()
