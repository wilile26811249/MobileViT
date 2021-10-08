from typing import List, Optional

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms as T

class AverageMeter(object):
    def __init__(self,
        name: str,
        fmt: Optional[str] = ':f',
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,
        val: float,
        n: Optional[int] = 1
    ) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: Optional[str] = "",
        batch_info: Optional[str] = ""
    ) -> None:
        self.batch_fmster = self._get_batch_fmster(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.batch_info = batch_info

    def display(self, batch):
        self.info = [self.prefix + self.batch_info + self.batch_fmster.format(batch)]
        self.info += [str(meter) for meter in self.meters]
        print('\t'.join(self.info))

    def _get_batch_fmster(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class EarlyStopping(object):
    """
    Arg
    """
    def __init__(self,
        patience: int = 7,
        verbose: Optional[bool] = False,
        delta: Optional[float] = 0.0,
        path: Optional[str] = "checkpoint.pt"
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop_flag = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.verbose = verbose
        self.path = path

    def __call__(self, val_loss, model):
        score = abs(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_model(val_loss, model)
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
                print(f"Best val loss: {self.val_loss_min}  Current val loss: {score}")
            if self.counter >= self.patience:
                self.early_stop_flag = True
        else:
            self.best_score = score
            self.save_model(val_loss, model)
            self.counter = 0

    def save_model(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def accuracy(output, target, topk = (1,)):
    """
    Computes the accuracy over the top k predictions
    """
    with torch.no_grad():
        max_k = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(max_k,
            dim = 1,
            largest = True,
            sorted = True
        )
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[: k].contiguous().view(-1).float().sum(0, keepdim = True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_cifar10_dataset(train_transform = None, test_transform = None):
    train_dataset =  datasets.CIFAR10(
        root = './data',
        train = True,
        transform = train_transform,
        download = True
    )
    test_dataset = datasets.CIFAR10(
        root = './data',
        train = False,
        transform = test_transform,
        download = True
    )
    return train_dataset, test_dataset


def get_dataloader(
        train_transform,
        test_transform,
        img_size = 224,
        split = (0.8, 0.2),
        **kwargs
    ):
    assert len(split) == 2
    assert sum(split) == 1
    assert split[0] + split[1] == 1

    train_dataset, test_dataset = get_cifar10_dataset(train_transform, test_transform)
    train_size = int(len(train_dataset) * split[0])
    test_size = int(len(train_dataset) * split[1])
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        (train_size, test_size)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = kwargs['batch_size'],
        shuffle = True,
        num_workers = kwargs['num_workers'],
        pin_memory = True,
        drop_last = True,
        sampler = None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = kwargs['batch_size'],
        shuffle = False,
        num_workers = kwargs['num_workers'],
        pin_memory = True,
        drop_last = False,
        sampler = None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = kwargs['batch_size'],
        shuffle = False,
        num_workers = kwargs['num_workers'],
        pin_memory = True,
        drop_last = False,
        sampler = None
    )
    return train_loader, val_loader, test_loader