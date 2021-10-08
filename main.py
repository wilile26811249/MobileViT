import os
import argparse
import utils
import shutil
import wandb
from tqdm import tqdm
import models

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.datasets as datasets


def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_epoch(epoch, net, train_loader, val_loader , criterion, optimizer, scheduler, device):
    """
    Training logic for an epoch
    """
    global best_acc1
    train_loss = utils.AverageMeter("Epoch losses", ":.4e")
    train_acc1 = utils.AverageMeter("Train Acc@1", ":6.2f")
    train_acc5 = utils.AverageMeter("Train Acc@5", ":6.2f")
    progress_train = utils.ProgressMeter(
        num_batches = len(val_loader),
        meters = [train_loss, train_acc1, train_acc5],
        prefix = 'Epoch: {} '.format(epoch + 1),
        batch_info = " Iter"
    )
    net.train()

    for it, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = utils.accuracy(outputs, targets, topk = (1, 5))

        train_loss.update(loss.item(), inputs.size(0))
        train_acc1.update(acc1.item(), inputs.size(0))
        train_acc5.update(acc5.item(), inputs.size(0))
        if it % args.print_freq == 0:
            progress_train.display(it)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log on Wandb
        wandb.log({
            "Loss/train" : train_loss.avg,
            "Acc@1/train" : train_acc1.avg,
            "Acc@5/train" : train_acc5.avg,
        })
    scheduler.step()

    # Validation model
    val_loss = utils.AverageMeter("Val losses", ":.4e")
    val_acc1 = utils.AverageMeter("Val Acc@1", ":6.2f")
    val_acc5 = utils.AverageMeter("Val Acc@5", ":6.2f")
    progress_val = utils.ProgressMeter(
        num_batches = len(val_loader),
        meters = [val_loss, val_acc1, val_acc5],
        prefix = 'Epoch: {} '.format(epoch + 1),
        batch_info = " Iter"
    )
    net.eval()

    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
        val_loss.update(loss.item(), inputs.size(0))
        val_acc1.update(acc1.item(), inputs.size(0))
        val_acc5.update(acc5.item(), inputs.size(0))
        acc1 = val_acc1.avg

        if it % args.print_freq == 0:
            progress_val.display(it)

        # Log on Wandb
        wandb.log({
            "Loss/val" : val_loss.avg,
            "Acc@1/val" : val_acc1.avg,
            "Acc@5/val" : val_acc5.avg
        })

    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    return val_loss.avg, val_acc1.avg, val_acc5.avg


if __name__ == "__main__":
    best_acc1 = 0.0
    parser = argparse.ArgumentParser(description = "Train classification of CMT model")
    parser.add_argument('--data', metavar = 'DIR', default = '../imagenet_data',
                help = 'path to dataset')
    parser.add_argument("--gpu_device", type = int, default = 2,
                help = "Select specific GPU to run the model")
    parser.add_argument('--batch-size', type = int, default = 256, metavar = 'N',
                help = 'Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type = int, default = 90, metavar = 'N',
                help = 'Number of epochs to train (default: 90)')
    parser.add_argument('--num-class', type = int, default = 1000, metavar = 'N',
                help = 'Number of classes to classify (default: 10)')
    parser.add_argument('--lr', type = float, default = 0.05, metavar='LR',
                help = 'Learning rate (default: 6e-5)')
    parser.add_argument('--weight-decay', type = float, default = 5e-5, metavar = 'WD',
                help = 'Weight decay (default: 1e-5)')
    parser.add_argument('-p', '--print-freq', default = 10, type = int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()

    # autotune cudnn kernel choice
    torch.backends.cudnn.benchmark = True

    # Create folder to save model
    WEIGHTS_PATH = "./weights"
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)

    # Set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
    ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True,
        num_workers = 4,  pin_memory = True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size, shuffle = False,
        num_workers = 4, pin_memory = True
    )

    # Create model
    net = models.MobileViT_S()
    # net.to(device)
    net = torch.nn.DataParallel(net).to(device)

    # Set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), args.lr,
                                momentum = 0.9,
                                weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Using wandb for logging
    wandb.init()
    wandb.config.update(args)
    wandb.watch(net)

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        loss, acc1, acc5 = train_epoch(epoch, net, train_loader,
            val_loader, criterion, optimizer, scheduler, device
        )
        print(f"Epoch {epoch} ->  Acc@1: {acc1}, Acc@5: {acc5}")

    print("Training is done")
