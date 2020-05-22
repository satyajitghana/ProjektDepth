from .base_trainer import BaseTrainer


class GPUTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(GPUTrainer, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch):

        return 0

    def test_epoch(self, epoch):

        return 0
