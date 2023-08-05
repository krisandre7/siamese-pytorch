import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from siamese import SiameseNetwork, SiameseDataset
from contrastive_loss import ContrastiveLoss
import utils
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--dataset_dir',
        type=str,
        help="Path to directory containing dataset.",
        default='dataset'
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for saving prediction images.",
        default='./eval'
    )
    parser.add_argument(
        '-c',
        '--checkpoint_dir',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )
    parser.add_argument(
        '-n',
        '--checkpoint_name',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        default='best.pt'
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    
    # Open config
    with open(os.path.join(args.checkpoint_dir, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Get image paths and labels from dataset
    file_paths, labels = utils.getDataset(args.dataset_dir)

    # Split image paths and labels using Stratified
    files_train, labels_train, files_test, labels_test = utils.stratifiedSortedSplit(
        file_paths, labels, config['train_size'], config['test_size'], config['random_seed'])
    
    train_count = np.unique(labels_train, return_counts=True)[1].mean()
    test_count = np.unique(labels_test, return_counts=True)[1].mean()
    print(
        f'Split {train_count} images from each class for train and {test_count} for test')

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load images into dataset
    val_dataset = SiameseDataset(
        files_test, labels_test, final_shape=config['final_shape'], **config['test_dataset'])

    val_dataloader = DataLoader(val_dataset, **config['val_dataloader'])

    # Instantiate model and move it to GPU
    model = SiameseNetwork(config['final_shape'][0], config['final_shape'][1], backbone=config['backbone'])
    model.to(device)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    loss_func = ContrastiveLoss(config['similarity_margin'])

    losses = []
    correct = 0
    total = 0

    inv_transform = transforms.Compose(
        [transforms.Resize([280, 320], antialias=None)])

    for i, (img1, img2, y, class1, class2) in enumerate(val_dataloader):
        print("[{} / {}]".format(i, len(val_dataloader)))

        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
        
        output1, output2 = model(img1, img2)
        loss = loss_func(output1, output2, y)

        losses.append(loss.item())
        correct += utils.similarityCorrect(y,
                                        output1, output2, config['similarity_margin'])
        total += len(y)

        fig = plt.figure("class1={}\tclass2={}".format(
            class1, class2), figsize=(4, 2))
        plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(
            class1, output1[0][0].item(), class2))

        # Apply inverse transform (denormalization) on the images to retrieve original images.
        img1 = inv_transform(img1).cpu().numpy()[0]
        img2 = inv_transform(img2).cpu().numpy()[0]
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(img1[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(img2[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the plot
        plt.savefig(os.path.join(args.out_path, '{}.png').format(i))

    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(
        sum(losses)/len(losses), correct / total))
