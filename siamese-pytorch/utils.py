import numpy as np
import glob
import os
import torch
from torch import Tensor
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F

def get_label(path: str):
    dataset_split = path.split('dataset',1)[1]
    label = int(dataset_split.split('/',2)[1])
    return label - 1

def getDataset(dataset_dir: str):
    file_paths = np.sort(glob.glob(dataset_dir + '/*/*.bmp'))
    labels = np.array([get_label(path) for path in file_paths], np.float32)
    
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

def similarityCorrect(y: Tensor, output1: Tensor, output2: Tensor, similarity_margin: float):
    """Calculate correct prediction for similarity scores

    Args:
        y (Tensor): ground truth labels
        y_pred (Tensor): predicted labels
        similarity_margin (float): threshold for measuring similarity predictions

    Returns:
        correct: number of correct label predictions
    """
    euclidean_distance = F.pairwise_distance(output1, output2)
    ones_index = torch.where(y == 1)[0]
    zeros_index = torch.where(y == 0)[0]
    
    true_positives = torch.count_nonzero(euclidean_distance[ones_index] < similarity_margin).item()
    true_negatives = torch.count_nonzero(euclidean_distance[zeros_index] > similarity_margin).item()
    # false_positives = torch.count_nonzero(y_pred[zeros_index] < similarity_margin).item()
    # false_negatives = torch.count_nonzero(y_pred[ones_index] > similarity_margin).item()
    
    # print(f'TP:{true_positives}, FP: {false_positives},\n TN: {true_negatives}, FN: {false_negatives}')
    return true_positives + true_negatives