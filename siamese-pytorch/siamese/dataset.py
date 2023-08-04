import os
import glob
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class SiameseDataset(torch.utils.data.IterableDataset):
    def __init__(self, image_paths: np.array, image_classes: Tensor, 
                 shuffle_pairs: bool, augment: bool, final_shape: tuple, grayscale: bool):

        self.image_paths = image_paths
        self.image_classes = image_classes

        num_channels = 1 if grayscale else 3
        self.shuffle_pairs = shuffle_pairs

        if augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                # transforms.RandomAffine(degrees=20, translate=(
                #     0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
                transforms.Resize(final_shape, antialias=None)
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
                transforms.Resize(final_shape, antialias=None)
            ])

        self.create_pairs()

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        # self.image_paths = glob.glob(os.path.join(self.path, "*/*.png"))
        # self.image_classes = []
        self.class_indices = {}

        for image_path, image_class in zip(self.image_paths, self.image_classes):

            if image_class not in self.class_indices:
                self.class_indices[int(image_class.item())] = []
            self.class_indices[int(image_class.item())].append(
                np.where(self.image_paths == image_path)[0])

        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        for i, pos in zip(self.indices1, select_pos_pair):
            class1 = self.image_classes[i]
            if pos:
                class2 = class1
            else:
                class2 = np.random.choice(
                    list(set(self.class_indices.keys()) - {class1}))
            idx2 = np.random.choice(self.class_indices[int(class2.item())][0])
            self.indices2.append(idx2)
        self.indices2 = np.array(self.indices2)

    def __iter__(self):
        self.create_pairs()

        for idx, idx2 in zip(self.indices1, self.indices2):

            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            image1 = self.transform(image1).float()
            image2 = self.transform(image2).float()

            yield (image1, image2), torch.FloatTensor([class1 == class2]), (class1, class2)

    def __len__(self):
        return len(self.image_paths)
