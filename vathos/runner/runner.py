from vathos.utils import setup_logger, get_instance_v2
import vathos.model as vmodel
import vathos.model.loss as vloss
import vathos.trainer as vtrainer
import vathos.data_loader as vdata_loader
from vathos.data_loader.utils import split_dataset
import vathos.utils as vutils

import torch
import torchvision.transforms as T
from pprint import pformat
import pprint
from pathlib import Path

logger = setup_logger(__name__)


class Runner():
    def __init__(self, config):
        self.config = config

        # print the super awesome logo
        print(vutils.logo)
        logger.info('Now simply setup_train and then start_train your model')

    def setup_train(self):

        cfg = self.config

        logger.info('Config')
        # print the config
        for line in pprint.pformat(cfg).split('\n'):
            logger.info(line)

        # dataset:
        #     name: DenseDepth
        #     root: vathos_data
        #     zip_dir: "/content/gdrive/My Drive/DepthProject/depth_dataset_zipped/"
        #     loader_args:
        #         batch_size: 128
        #         num_workers: 4
        #         shuffle: True
        #         pin_memory: True
        dataset = get_instance_v2(
            vdata_loader,
            cfg['dataset']['name'],
            root=cfg['dataset']['root'],
            source_zipfolder=cfg['dataset']['zip_dir'],
            transform=T.Compose([T.ToTensor()]),
            target_transform=T.Compose([T.ToTensor()])
        )

        train_subset, test_subset = split_dataset(dataset)

        # check if the train_subset and test_subset indices are present in disk
        subset_file = Path(cfg['dataset']['root']) / 'subset.pt'

        if subset_file.exists():
            # load the subset state
            logger.info('=> Found subset.pt loading indices')
            subset_state = torch.load(subset_file)

            train_subset.indices = subset_state['train_indices']
            test_subset.indices = subset_state['test_indices']
        else:
            # save the subset dict
            torch.save({'train_indices': train_subset.indices,
                        'test_indices': test_subset.indices}, subset_file)
            logger.info('=> Saved subset.pt (train, test indices)')

        # create the model
        model = get_instance_v2(vmodel, cfg['model'])

        # optimizer:
        #     type: AdamW
        #     args:
        #         lr: 0.01
        optimizer = get_instance_v2(torch.optim, ctor_name=cfg['optimizer']['type'], params=model.parameters(
        ), lr=cfg['optimizer']['args']['lr'])

        # seg_loss: BCEDiceLoss
        # depth_loss: RMSELoss
        seg_loss = get_instance_v2(vloss, ctor_name=cfg['seg_loss'])
        depth_loss = get_instance_v2(vloss, ctor_name=cfg['depth_loss'])

        loss_fns = (seg_loss, depth_loss)

        # check if the model init weights are specified
        # model_init: "models/model.pt"
        model_init = Path(cfg['model_init'])
        if model_init.exists():
            logger.info('=> Found Model init weights')
            model_state_dict = torch.load(model_init)
            model.load_state_dict(model_state_dict)

        # load the last checkpoint
        # chkpt_dir: checkpoint
        model_checkpoint = Path(cfg['chkpt_dir']) / 'model_checkpoint.pt'
        train_checkpoint = Path(cfg['chkpt_dir']) / 'train_checkpoint.pt'

        if model_checkpoint.exists():
            logger.info('=> Found model checkpoint')
            model_state_dict = torch.load(model_checkpoint)
            model.load_state_dict(model_state_dict)

        state_dict = None
        if train_checkpoint.exists():
            logger.info('=> Found train checkpoint')
            checkpoint_state = torch.load(train_checkpoint)

            optimizer.load_state_dict(checkpoint_state['optimizer'])
            save_epoch = checkpoint_state['save_epoch']
            total_epochs = checkpoint_state['total_epochs']
            logger.info(f'Start Epoch should be {save_epoch}+1')

            state_dict = checkpoint_state

        else:
            logger.info('=> No saved checkpoints found')

        if cfg['device'] == 'GPU':
            self.trainer = get_instance_v2(
                vtrainer, 'GPUTrainer', model, loss_fns, optimizer, cfg, train_subset, test_subset, state_dict=state_dict)
        else:
            logger.error(f"Unsupported Device: {cfg['device']}")
