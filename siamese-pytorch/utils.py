import numpy as np
import glob
import os
import torch
from torch import Tensor
from sklearn.model_selection import StratifiedShuffleSplit

def getDataset(dataset_dir: str):
    file_paths = np.sort(glob.glob(dataset_dir + '/*/*.bmp'))

    labels = np.array([path.split('/')[2] for path in file_paths], np.float32)
    labels -= 1
    
    return file_paths, labels

def stratifiedSortedSplit(file_paths: np.array, labels: np.array, 
                    train_size: float, test_size: float, random_state: int):
    """Splits image paths and labels equally for each class, then sorts them"""
    splitter = StratifiedShuffleSplit(n_splits=1, 
                                      train_size=train_size, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(file_paths, labels))
    
    files_train, labels_train = file_paths[train_indices], labels[train_indices]
    files_test, labels_test = file_paths[test_indices], labels[test_indices]

    sort_index = np.argsort(labels_train)
    labels_train = labels_train[sort_index]
    files_train = files_train[sort_index]

    sort_index = np.argsort(labels_test)
    labels_test = labels_test[sort_index]
    files_test = files_test[sort_index]

    labels_train: Tensor = torch.from_numpy(labels_train)
    labels_test: Tensor = torch.from_numpy(labels_test)
    
    return files_train, labels_train, files_test, labels_test