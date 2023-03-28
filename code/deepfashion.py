import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
import ipdb
import cv2
from PIL import Image
from torch.utils.data import Dataset
#from .utils import check_integrity, download_and_extract_archive
#from .vision import VisionDataset


class DeepFashion(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        label_transform (callable, optional): A function/transform that takes in the
            label and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.split = split # training set, val set or test set
        self.transform = transform
        self.label_transform = label_transform
        
        img_path_file = os.path.join(self.root, 'split', f'{self.split}.txt')
        label_file = os.path.join(self.root, 'split', f'{self.split}_attr.txt')
        
        self.imgs = self.load_images(self.root, img_path_file)
        self.labels = self.load_labels(label_file)
        
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_images(self, img_root, img_path_file):
        
        file = open(img_path_file, 'r')
        img_path_ls = file.read().splitlines()
        
        img_paths = [os.path.join(img_root, fname) for fname in img_path_ls]
        
        imgs = []
        for img_path in img_paths:
            im = Image.open(img_path)
            im = np.asarray(im)
            im = cv2.resize(im, (256, 256))
            imgs.append(im)
        
        #ipdb.set_trace()
        imgs = np.stack(imgs, 0)
        return imgs
        
    
    def load_labels(self, label_file):
        
        if self.split == 'test':
            return np.zeros((1000,6))
        file = open(label_file, 'r')
        data = file.read().splitlines()
        data = [line.split(' ') for line in data]
        
        def str2int(line):
            return np.array([int(ele) for ele in line])
        data = [str2int(line) for line in data]
        labels = np.stack(data, 0)
        return labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where label is index of the label class.
        """
        img, label = self.imgs[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self) -> int:
        return self.imgs.shape[0]
