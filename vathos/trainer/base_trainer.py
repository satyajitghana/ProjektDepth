import abc


class BaseTrainer(metaclass=abc.ABCMeta):
    r'''
    BaseTrainer: An Abstract Meta Class for all trainers (GPU, CPU, TPU)
    '''

    def __init__(self, model, lossfn, optimizer, config, train_loader, test_loader, lr_scheduler):
        super(BaseTrainer, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.epochs = self.config['training']['epochs']

    @abc.abstractclassmethod
    def train_epoch(self, epoch):
        pass

    @abc.abstractclassmethod
    def test_epoch(self, epoch):
        pass
