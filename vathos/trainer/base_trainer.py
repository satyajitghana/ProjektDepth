import abc


class BaseTrainer(metaclass=abc.ABCMeta):
    r'''
    BaseTrainer: An Abstract Meta Class for all trainers (GPU, CPU, TPU)
    '''

    def __init__(self, model, loss_fns, optimizer, config, train_subset, test_subset):
        super(BaseTrainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.seg_loss, self.depth_loss = loss_fns
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.epochs = self.config['training']['epochs']

        self.comb_loss = lambda l1, l2: l1 + 2*l2

    @abc.abstractclassmethod
    def train_epoch(self, epoch):
        pass

    @abc.abstractclassmethod
    def test_epoch(self, epoch):
        pass
