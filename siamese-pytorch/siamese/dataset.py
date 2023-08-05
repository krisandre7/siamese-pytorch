import time

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torchvision import transforms
import utils

class SiameseDataset(torch.utils.data.IterableDataset):
    def __init__(self, image_paths: np.array, image_classes: Tensor, augment: bool, final_shape: tuple, grayscale: bool, shuffle: bool):

        self.image_paths = image_paths
        self.image_classes = image_classes
        
        self.unique_classes = torch.unique(image_classes).numpy().astype(int)

        self.grayscale = grayscale
        self.shuffle = shuffle

        if augment:
            # If images are to be augmented, add extra operations for it (first two).
            self.transform = transforms.Compose([
                # transforms.RandomAffine(degrees=20, translate=(
                #     0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                #                      0.229, 0.224, 0.225]),
                transforms.Resize(final_shape[1:], antialias=None)
            ])
        else:
            # If no augmentation is needed then apply only the normalization and resizing operations.
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                #                      0.229, 0.224, 0.225]),
                transforms.Resize(final_shape[1:], antialias=None)
            ])

        self.images_dict = {
            image_class: 
                [image_path for image_path in self.image_paths if utils.get_label(image_path) == image_class]
            for image_class in self.unique_classes
        }
        
    def __iter__(self):
        # self.create_pairs()
        
        np.random.seed(int(time.time())) if self.shuffle else np.random.seed(42) 

        for image_class, image_paths in self.images_dict.items():

            label = np.random.randint(0,2)
            
            size = len(image_paths)
            
            if label and size > 1:
                
                index1 = np.random.randint(0, size)
                
                index2 = np.random.randint(0, size)
                while index1 == index2:
                    index2 = np.random.randint(0, size)
                
                image_path1 = image_paths[index1]
                image_path2 = image_paths[index2]  
                class1 = image_class.item()
                class2 = class1
                        
            else:
                index1 = np.random.randint(0, size)
                
                rand_class = np.random.choice(self.unique_classes)
                
                while rand_class == image_class:
                    rand_class = np.random.choice(self.unique_classes)
                    
                rand_image_paths = self.images_dict[rand_class]
                
                index2 = np.random.randint(0, len(rand_image_paths))
                
                image_path1 = image_paths[index1]
                image_path2 = rand_image_paths[index2]
                class1 = image_class.item()
                class2 = rand_class
                
            image1 = Image.open(image_path1).convert("L" if self.grayscale == 1 else "RGB")
            image2 = Image.open(image_path2).convert("L" if self.grayscale == 1 else "RGB")

            image1 = self.transform(image1).float()
            image2 = self.transform(image2).float()
            
            yield image1, image2, torch.FloatTensor([label]), class1, class2

    def __len__(self):
        return len(self.image_paths)
