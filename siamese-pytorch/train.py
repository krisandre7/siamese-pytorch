import utils
from siamese import SiameseNetwork, SiameseDataset
from contrastive_loss import ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import yaml
import os
import randomname
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_loop(model: nn.Module, loss_func: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()

    losses = []
    correct = 0
    total = 0

    for img1, img2, y, class1, class2 in train_dataloader:
        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

        output1, output2 = model(img1, img2)
        loss = loss_func(output1, output2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        correct += utils.similarityCorrect(y,
                                           output1, output2, args['similarity_margin'])
        total += len(y)

    train_loss = sum(losses)/len(losses)
    accuracy = (correct / total) * 100
    
    return train_loss, accuracy

def test_loop(model: nn.Module, loss_func: nn.Module, optimizer: torch.optim.Optimizer, best_val: float):
    model.eval()

    losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, y, class1, class2 in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            image_embed1, image_embed2 = model(img1, img2)
            loss = loss_func(image_embed1, image_embed2, y)

            losses.append(loss.item())

            correct += utils.similarityCorrect(y,
                                               image_embed1, image_embed2, args['similarity_margin'])
            total += len(y)

    val_loss = sum(losses)/max(1, len(losses))

    accuracy = 100 * correct / total
    
    return val_loss, accuracy
    # Evaluation Loop End


# baseado em https://github.com/sohaib023/siamese-pytorch
if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Open config
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)

    # Assign a unique folder as training output
    training_id = f'{randomname.get_name()}-{str(random.randint(1,9))}'

    final_path = os.path.join(
        args['output_path'], training_id)

    while os.path.isdir(final_path):
        training_id = f'{randomname.get_name()}-{str(random.randint(1,9))}'
        final_path = os.path.join(
            args['output_path'], training_id)

    os.makedirs(final_path, exist_ok=True)

    # Write config to output folder
    with open(os.path.join(final_path, 'config.yaml'), 'w') as file:
        yaml.dump(args, file)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get image paths and labels from dataset
    dataset_path = os.path.join(args['dataset_dir'], args['dataset_name'])
    file_paths, labels = utils.getDataset(dataset_path)

    # Split image paths and labels using Stratified
    files_train, labels_train, files_test, labels_test = utils.stratifiedSortedSplit(
        file_paths, labels, args['train_size'], args['test_size'], args['random_seed'])

    train_count = np.unique(labels_train, return_counts=True)[1].mean()
    test_count = np.unique(labels_test, return_counts=True)[1].mean()
    print(
        f'Split {train_count} images from each class for train and {test_count} for test')

    # Load images into dataset
    train_dataset = SiameseDataset(
        files_train, labels_train, final_shape=args['final_shape'], **args['train_dataset'])
    val_dataset = SiameseDataset(
        files_test, labels_test, final_shape=args['final_shape'], **args['test_dataset'])

    train_dataloader = DataLoader(train_dataset, **args['train_dataloader'])
    val_dataloader = DataLoader(val_dataset, **args['val_dataloader'])

    # Instantiate model and move it to GPU
    model = SiameseNetwork(args['final_shape'][0], args['final_shape'][1], backbone=args['backbone'])
    model.to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    loss_func = ContrastiveLoss(args['similarity_margin'])

    # Initialize TensorBoard
    writer = SummaryWriter(os.path.join(final_path))

    best_val = 10000000000
    for epoch in range(args['epochs']):
        print("[{} / {}]".format(epoch+1, args['epochs']))

        train_loss, accuracy = train_loop(model, loss_func, optimizer)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', accuracy, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}%\t".format(
            train_loss, accuracy))
        val_loss, accuracy = test_loop(model, loss_func, optimizer, best_val)
        
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', accuracy, epoch)
        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}%\t".format(
            val_loss, accuracy))
        
        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if args['save_best'] and val_loss < best_val:
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
        if args['save_after'] != 0 and (epoch + 1) % args['save_after'] == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args['backbone'],
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(final_path, "epoch_{}.pt".format(epoch + 1))
            )
