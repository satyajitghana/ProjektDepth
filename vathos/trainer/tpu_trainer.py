from .base_trainer import BaseTrainer

import torch
import os


class TPUTrainer(BaseTrainer):
    r"""TPUTrainer: Trains the vathos model on TPU
    """

    def __init__(self, *args, **kwargs):

        super(TPUTrainer, self).__init__(*args, **kwargs)
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp

        os.environ['XLA_USE_BF16'] = 1

    @staticmethod
    def _xla_train(index):
        device = xm.xla_device()
        para_loader = pl.ParallelLoader(train_loader, [device])

        model = MNIST().train().to(device)
        loss_fn = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        for data, target in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)

    def train_epoch(self, epoch):

        return 0

    def test_epoch(self, epoch):

        device = xm.xla_device()

        xm.mark_step()

        return 0
