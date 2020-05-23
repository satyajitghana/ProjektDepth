from pathlib import Path
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import PIL
import zipfile
from zipfile import ZipFile
sns.set()


class DenseDepth(Dataset):
    '''
    DenseDepth Dataset

    Args:
        root: the directory where to place the dataset unzipped files
        source_zipfiles: the directory where the dataset zip files are stored (mount your drive and give a absolute path)
        transform: torchvision transform for input images
        target_transform: torchvision transform for ouput images

    Input is fg_bg image AND bg image
    Target is fg_bg_mask AND depth_fg_bg image

    '''

    source_zipfiles = ['bg_small.zip', 'fg_bg_small.zip',
                       'fg_bg_mask_small.zip', 'depth_fg_bg_small.zip']

    bg_stat = (['0.573435604572296', '0.520844697952271', '0.457784473896027'], [
               '0.207058250904083', '0.208138316869736', '0.215291306376457'])
    fg_bg_stat = (['0.568499565124512', '0.512103974819183', '0.452332496643066'], [
                  '0.211068645119667', '0.211040720343590', '0.216081097722054'])
    fg_bg_mask_stat = (['0.062296919524670', '0.062296919524670', '0.062296919524670'], [
                       '0.227044790983200', '0.227044790983200', '0.227044790983200'])
    depth_fg_bg_stat = (['0.302973538637161', '0.302973538637161', '0.302973538637161'], [
                        '0.101284727454185', '0.101284727454185', '0.101284727454185'])

    def __init__(self, root, source_zipfolder, train=True, transform=None, target_transform=None):
        self.root = Path(root) / 'Vathos'
        self.root.mkdir(parents=True, exist_ok=True)
        self.source_zipfolder = Path(source_zipfolder)
        self.transform = transform
        self.target_transform = target_transform

        # check if the dataset exists
        if os.path.isdir(self.root / 'bg') or os.path.isdir(self.root / 'fg_bg') or os.path.isdir(self.root / 'fg_bg_mask') or os.path.isdir(self.root / 'depth_fg_bg'):
            print(f'dataset folders/files already exists in {self.root}')
        else:
            # extract the dataset to root dir
            self.extractall()

        # pathlib does not order them by default
        bg_paths = sorted(list(Path(self.root / 'bg').glob('*.jpg')))
        fg_bg_paths = sorted(list(Path(self.root / 'fg_bg').glob('**/*.jpg')))
        fg_bg_mask_paths = sorted(
            list(Path(self.root / 'fg_bg_mask').glob('**/*.jpg')))
        depth_fg_bg_paths = sorted(
            list(Path(self.root / 'depth_fg_bg').glob('**/*.png')))

        assert(len(bg_paths) == 100)
        assert(len(fg_bg_paths) == 400000)
        assert(len(fg_bg_mask_paths) == 400000)
        assert(len(depth_fg_bg_paths) == 400000)

        print(f'found {len(bg_paths)} bg images, {len(fg_bg_paths)} fg_bg images, {len(fg_bg_mask_paths)} fg_bg_mask images, {len(depth_fg_bg_paths)} depth_fg_bg images')

        self.input_paths = fg_bg_paths
        self.bg_paths = bg_paths
        self.target_paths = list(zip(fg_bg_mask_paths, depth_fg_bg_paths))

    def extractall(self):
        print(f'Extracting the zip files')
        for smallzip in tqdm(self.source_zipfiles):
            print(f'Extracting {smallzip} ...')
            zipf = ZipFile(self.source_zipfolder / smallzip, 'r')
            zipf.extractall(self.root)

    def __getitem__(self, index):

        bgidx = self.input_paths[index].stem.split('_')[3]

        bgimg = Image.open(self.bg_paths[int(bgidx)])
        bgimg = bgimg.convert('RGB')
        # bgimg = np.array(bgimg)

        fg_bgimg = Image.open(self.input_paths[index])
        fg_bgimg = fg_bgimg.convert('RGB')
        # fg_bgimg = np.array(fg_bgimg)

        target_mask, target_depth = self.target_paths[index]

        mask_fg_bgimg = Image.open(target_mask)
        mask_fg_bgimg.convert('L')
        mask_arr = np.array(mask_fg_bgimg)
        mask_arr[mask_arr >= 150] = 255
        mask_arr[mask_arr < 150] = 0
        mask_fg_bgimg = Image.fromarray(mask_arr)
        # mask_fg_bgimg.convert('L')

        depth_fg_bgimg = Image.open(target_depth)
        depth_fg_bgimg.convert('L')

        if self.transform is not None:
            bgimg = self.transform(bgimg)
            fg_bgimg = self.transform(fg_bgimg)

        if self.target_transform is not None:
            mask_fg_bgimg = self.target_transform(mask_fg_bgimg)
            depth_fg_bgimg = self.target_transform(depth_fg_bgimg)

        return {'bg': bgimg, 'fg_bg': fg_bgimg, 'fg_bg_mask': mask_fg_bgimg, 'depth_fg_bg': depth_fg_bgimg}

    def __len__(self):
        return len(self.input_paths)

    @staticmethod
    def plot_sample(sample):
        '''
        Plots a given sample of the dataset
        '''
        bg, fg_bg, fg_bg_mask, depth_fg_bg = sample['bg'].permute(1, 2, 0).numpy(), sample['fg_bg'].permute(
            1, 2, 0).numpy(), sample['fg_bg_mask'][0].numpy(), sample['depth_fg_bg'][0].numpy()
        fig, ax = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)

        ax[0, 0].imshow(bg)
        ax[0, 0].axis('off')

        ax[0, 1].imshow(fg_bg)
        ax[0, 1].axis('off')

        ax[1, 0].imshow(fg_bg_mask)
        ax[1, 0].axis('off')

        ax[1, 1].imshow(depth_fg_bg)
        ax[1, 1].axis('off')

        fig.tight_layout()

        return fig

    @staticmethod
    def plot4_batch(batch):
        '''
        Plots 4 images for batch
        '''
        fig, ax = plt.subplots(4, 4, figsize=(6, 6), sharex=True, sharey=True)

        # set the title
        for axs, col in zip(ax[0], ['BG', 'FG_BG', 'FG_BG_MASK', 'DEPTH_FG_BG']):
            axs.set_title(col)

        # plot the first 4 samples from the batch
        for i in range(4):
            bg, fg_bg, fg_bg_mask, depth_fg_bg = batch['bg'][i].permute(1, 2, 0).cpu().numpy(), batch['fg_bg'][i].permute(
                1, 2, 0).cpu().numpy(), batch['fg_bg_mask'][i][0].cpu().numpy(), batch['depth_fg_bg'][i][0].cpu().numpy()

            ax[i, 0].imshow(bg)
            ax[i, 0].axis('off')

            ax[i, 1].imshow(fg_bg)
            ax[i, 1].axis('off')

            fg_bg_mask[fg_bg_mask >= 0.9] = 1
            fg_bg_mask[fg_bg_mask < 0.9] = 0

            ax[i, 2].imshow(fg_bg_mask)
            ax[i, 2].axis('off')

            ax[i, 3].imshow(depth_fg_bg)
            ax[i, 3].axis('off')

        fig.tight_layout()

        return fig

    @staticmethod
    def plot_results(batch):
        '''
        Plots 4 images for batch's model results
        '''
        fig, ax = plt.subplots(4, 6, figsize=(10, 6), sharex=True, sharey=True)

        # set the title
        for axs, col in zip(ax[0], ['BG', 'FG_BG', 'GT MASK', 'PRED MASK', 'GT DEPTH', 'PRED DEPTH']):
            axs.set_title(col)

        # plot the first 4 samples from the batch
        for i in range(4):
            bg, fg_bg, fg_bg_mask, depth_fg_bg = batch['bg'][i].permute(1, 2, 0).cpu().numpy(), batch['fg_bg'][i].permute(
                1, 2, 0).cpu().numpy(), batch['fg_bg_mask'][i][0].cpu().numpy(), batch['depth_fg_bg'][i][0].cpu().numpy()

            pred_mask, pred_depth = batch['pred_mask'][i][0].cpu(
            ).numpy(), batch['pred_depth'][i][0].cpu().numpy()
            pred_mask[pred_mask >= 0.9] = 1
            pred_mask[pred_mask < 0.9] = 0

            ax[i, 0].imshow(bg)
            ax[i, 0].axis('off')

            ax[i, 1].imshow(fg_bg)
            ax[i, 1].axis('off')

            ax[i, 2].imshow(fg_bg_mask)
            ax[i, 2].axis('off')

            ax[i, 3].imshow(pred_mask)
            ax[i, 3].axis('off')

            ax[i, 4].imshow(depth_fg_bg)
            ax[i, 4].axis('off')

            ax[i, 5].imshow(pred_depth)
            ax[i, 5].axis('off')

        fig.tight_layout()

        return fig

    @staticmethod
    def apply_on_batch(batch, apply_func):
        batch['bg'] = apply_func(batch['bg'])
        batch['fg_bg'] = apply_func(batch['fg_bg'])
        batch['fg_bg_mask'] = apply_func(batch['fg_bg_mask'])
        batch['depth_fg_bg'] = apply_func(batch['depth_fg_bg'])

        return batch
