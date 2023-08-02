import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import yaml
from time import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from siamese import SiameseNetwork
from dataset import Dataset

import utils

# baseado em https://github.com/sohaib023/siamese-pytorch
if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)

    training_id = f'{time():.0f}'
    final_path = os.path.join(
        args['output_path'], args['dataset_name'], args['model_name'], training_id)
    os.makedirs(final_path, exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = os.path.join(args['dataset_dir'], args['dataset_name'])
    file_paths, labels = utils.getDataset(dataset_path)
    files_train, labels_train, files_test, labels_test = utils.trainTestSplit(
        file_paths, labels)

    train_dataset = Dataset(files_train, labels_train, args['final_shape'],
                            args['grayscale'], shuffle_pairs=True, augment=False)
    val_dataset = Dataset(files_test, labels_test, args['final_shape'],
                          args['grayscale'], shuffle_pairs=False, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = SiameseNetwork(backbone=args['backbone'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(final_path, "summary"))

    best_val = 10000000000

    for epoch in range(args['epochs']):
        print("[{} / {}]".format(epoch, args['epochs']))
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), y, (class1, class2) in train_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        train_loss = sum(losses)/len(losses)
        accuracy = (correct / total) * 100
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', accuracy, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(
            train_loss, accuracy))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2) in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        val_loss = sum(losses)/max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        accuracy = 100 * correct / total
        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, accuracy))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args['backbone'],
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(final_path, "best.pt")
            )

        # Save model based on the frequency defined by "args['save_after']"
        if (epoch + 1) % args['save_after'] == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args['backbone'],
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(final_path, "epoch_{}.pt".format(epoch + 1))
            )
