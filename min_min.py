import torch

from networks import *
from data import get_train_loader, get_test_loader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import os
from utils import *
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('CIFAR10-MIN-MIN')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--epsilon', default=8/255, type=float)
    parser.add_argument('--alpha', default=2/255, type=float)
    parser.add_argument('--steps', default=7, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=(30, 45), type=tuple[int])
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--train_total', default=50000, type=int)
    parser.add_argument('--interval', default=20, type=int)
    parser.add_argument('--stop-error', default=0.01, type=float)
    return parser.parse_args()


class MinMin(object):
    def __init__(self):
        self.model, self.optimizer, self.scheduler = None, None, None
        self.perts = torch.zeros((args.train_total, 3, 32, 32))

    def run(self):
        self.model_init()
        self.min_min()
        self.retrain()

    def model_init(self):
        self.model = eval(model_name)(num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=args.milestones, gamma=args.gamma)

    def update_pert(self):
        perts = []
        indices = []
        for data, target, index in train_loader:
            data, target = data.to(args.device), target.to(args.device)
            adv_data = pgd_inf(self.model, data, target, args.epsilon, args.alpha, args.steps, args.random_start,
                               reverse_direction=True)
            pert = adv_data - data
            perts.append(pert.cpu())
            indices.append(index)
        perts = torch.cat(perts, dim=0)
        indices = torch.cat(indices, dim=0)
        order = indices.sort()[1]
        perts = perts[order]
        self.perts = perts

    def min_min(self):
        num_batches = 1
        for epoch in range(args.epochs):
            self.model.train()
            loss_list = []

            for batch_idx, (data, target, index) in enumerate(train_loader):
                self.model.train()
                num_batches += 1
                data += self.perts[index]
                data, target = data.to(args.device), target.to(args.device)
                loss = F.cross_entropy(self.model(data), target)
                self.optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item()*len(data))
                self.optimizer.step()
                print(epoch, batch_idx, num_batches, num_batches%args.interval)
                if num_batches % args.interval == 0:
                    print('yes')
                    self.update_pert()
                    test_error = self.test(test_loader)
                    if test_error <= args.stop_error:
                        break

            else:
                avg_loss = np.array(loss_list).sum().item()/args.train_total
                train_acc = self.test(train_loader)
                test_acc = self.test(test_loader)
                logger.add_scalar('minmin_avg_loss', avg_loss, global_step=epoch)
                logger.add_scalar('minmin_test_acc', test_acc, global_step=epoch)
                logger.add_scalar('minmin_train_acc', train_acc, global_step=epoch)
                self.scheduler.step()
                continue
            break

    def test(self, loader):
        self.model.eval()
        total = 0
        correct = 0
        for data, target, index in loader:
            data, target = data.to(args.device), target.to(args.device)
            total += len(data)
            with torch.no_grad():
                correct += self.model(data).argmax(1).eq(target).sum().item()
        acc = correct / total
        return acc

    def retrain(self):
        self.model_init()
        for epoch in range(args.epochs):
            self.model.train()
            loss_list = []
            total = 0
            for data, target, index in train_loader:
                total += len(data)
                data += self.perts[index]
                data, target = data.to(args.device), target.to(args.device)
                loss = F.cross_entropy(self.model(data), target)
                self.optimizer.zero_grad()
                loss.backward()
                loss_list.append(loss.item() * len(data))
                self.optimizer.step()
            self.scheduler.step()
            avg_loss = np.array(loss_list).sum().item()/total
            train_acc = self.test(train_loader)
            test_acc = self.test(test_loader)
            logger.add_scalar('retrain_avg_loss', avg_loss, global_step=epoch)
            logger.add_scalar('retrain_test_acc', test_acc, global_step=epoch)
            logger.add_scalar('retrain_train_acc', train_acc, global_step=epoch)


if __name__ == '__main__':
    args = get_args()
    model_name = 'resnet18'
    test_loader = get_test_loader(args.batch_size)
    train_loader = get_train_loader(args.batch_size)
    path = os.path.join('logs/min_min', model_name)
    path = os.path.join(path, f'epochs={args.epochs}')
    os.makedirs(path, exist_ok=True)
    logger = SummaryWriter(log_dir=path)
    logger.add_text('args', str(args))

    minmin = MinMin()
    minmin.run()

