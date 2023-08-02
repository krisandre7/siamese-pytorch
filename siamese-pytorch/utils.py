import numpy as np
import glob
import os
import torch
from sklearn.model_selection import StratifiedShuffleSplit

def getDataset(dataset_dir: str):
    # class_names = np.sort(np.array(next(os.walk(DATASET_DIR))[1], np.float32))
    file_paths = np.sort(glob.glob(dataset_dir + '/*/*.bmp'))

    labels = np.array([path.split('/')[2] for path in file_paths], np.float32)
    labels -= 1
    
    return file_paths, labels

def trainTestSplit(file_paths, labels):
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=42)
    train_indices, test_indices = next(splitter.split(file_paths, labels))
    
    files_train, labels_train = file_paths[train_indices], labels[train_indices]
    files_test, labels_test = file_paths[test_indices], labels[test_indices]

    sort_index = np.argsort(labels_train)
    labels_train = labels_train[sort_index]
    files_train = files_train[sort_index]

    sort_index = np.argsort(labels_test)
    labels_test = labels_test[sort_index]
    files_test = files_test[sort_index]

    labels_train = torch.from_numpy(labels_train)
    labels_test = torch.from_numpy(labels_test)
    
    return files_train, labels_train, files_test, labels_test