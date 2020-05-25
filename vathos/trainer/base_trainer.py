import abc
import os
from pathlib import Path
from datetime import datetime

from vathos.utils import get_instance_v2, setup_logger

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.utils as utils
import torch.optim as optim

logger = setup_logger(__name__)


def optimizer_to(optim, device):
    r"""moves the optimizer to device

    Args:
        optim: the optimizer
        device: device to which to move to
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(sched, device):
    r"""moves the scheduler to device

    Args:
        sched: the scheduler
        device: device to which to move to
    """
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)


class BaseTrainer(metaclass=abc.ABCMeta):
    r"""BaseTrainer: An Abstract Meta Class for all trainers (GPU, CPU, TPU)

    Args:
        model: the model to be trained, (can be on cpu/gpu)
        loss_fns (Tuple): (seg_loss, depth_loss) 
        optimizer: the optimizer (can be on cpu/gpu)
        config: config in dict format
        train_subset(torch.utils.data.Subset): train dataset wrapped in a subset containing the indices
        test_subset(torch.utils.data.Subset): test dataset wrapped in a subset containing the indices
        state_dict(Optional): the saved state in a dictionary format
    """

    def __init__(self, model, loss_fns, optimizer, config, train_subset, test_subset, state_dict=None):
        super(BaseTrainer, self).__init__()

        cfg = config

        self.model = model
        self.optimizer = optimizer
        self.seg_loss, self.depth_loss = loss_fns
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.config = config
        self.start_epoch = 0

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.writer = SummaryWriter(
            f"{config['log_dir']}/{config['name']}.{current_time}")

        self.comb_loss = lambda l1, l2: l1 + 2*l2

        self.train_loader = utils.data.DataLoader(self.train_subset)
        self.train_loader = get_instance_v2(
            utils.data, 'DataLoader', self.train_subset, **cfg['dataset']['loader_args'])
        self.test_loader = get_instance_v2(
            utils.data, 'DataLoader', self.test_subset, **cfg['dataset']['loader_args'])

        if state_dict is not None:
            self.start_epoch = state_dict['save_epoch']+1

        self.epochs = self.config['training']['epochs'] - self.start_epoch

        # best accuracy
        self.best_accuracy = {'mrmse': 1e5, 'miou': 0}

        # lr_scheduler:
        #     type: OneCycleLR
        #     args:
        #         max_lr: 0.6
        if cfg['lr_scheduler']['type'] == 'OneCycleLR':
            self.lr_scheduler = get_instance_v2(optim.lr_scheduler, cfg['lr_scheduler']['type'], optimizer=self.optimizer, steps_per_epoch=len(
                self.train_loader), epochs=self.epochs, **cfg['lr_scheduler']['args'])
        else:
            self.lr_scheduler = get_instance_v2(
                optim.lr_scheduler, cfg['lr_scheduler']['type'], optimizer=self.optimizer, **cfg['lr_scheduler']['args'])

        if state_dict is not None:
            if 'scheduler' in state_dict:
                logger.info('Found lr_scheduler state')
                self.lr_scheduler.load_state_dict(state_dict['scheduler'])
            self.best_accuracy = state_dict['best_accuracy']

    @abc.abstractclassmethod
    def train_epoch(self, epoch):
        pass

    @abc.abstractclassmethod
    def test_epoch(self, epoch):
        pass
