"""
A place to test a bunch of different functions
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from torchvision import transforms, models as torchmodels, datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from utils import load_split_train_test

batch_size = 128
imgsize = (128, 128)
root = 'data/'
train_loader, test_loader = load_split_train_test(root, batch_size=batch_size, size=imgsize)

class Hierarchical:
    'A hierarchical classifier'
    def __init__(self, num_classes, epochs, batch_size, learning_rate, loaders):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.resnets = []
        self.train_loader, self.test_loader = loaders
        for i in range(num_classes):
            self.resnets.append(torchmodels.resnet50(num_classes=2))
            self.resnets[i].apply(self.init_weights)
            # if torch.cuda.is_available():
            #    self.resnets[i] = self.resnets[i].cuda()

        # Optimizers, schedulers, and criterion
        self.optimizers, self.schedulers = [], []
        for i in range(num_classes):
            self.optimizers.append(optim.Adam(self.resnets[i].parameters(), lr=learning_rate))
            self.schedulers.append(lr_scheduler.StepLR(self.optimizers[i], step_size=2, gamma=0.5))

        self.criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()

    def init_weights(self, m):
        'Kaiming weights'
        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def change_labels(self, labels, idx):
        for i in range(len(labels)):
            if labels[i] != idx:
                labels[i] = 0
            else:
                labels[i] = 1
        return labels

    def train_idx(self, idx, epoch, train_loader):
        'Trains a net for 1 epoch'
        print('Training model id: {}\tEpoch: {}'.format(idx, epoch))
        start_time = time.time()
        if torch.cuda.is_available():
            self.resnets[idx] = self.resnets[idx].cuda()
        self.resnets[idx].train()
        self.schedulers[idx].step()
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Change multi-class to binary
            labels = self.change_labels(labels, idx)
            features, labels = Variable(features), Variable(labels)
            if torch.cuda.is_available():
                features, labels = features.cuda(), labels.cuda()
            self.optimizers[idx].zero_grad()
            output = self.resnets[idx](features)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizers[idx].step()
            if (batch_idx + 1) % 50 == 0:
                batch_size = len(features)
                print('Train epoch {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, (batch_idx + 1) * len(features), len(train_loader) * batch_size,
                           100. * (batch_idx + 1) / len(train_loader), loss.item()))
        # Bring it back to where it came from
        self.resnets[idx].cpu()
        print('--- Time to train model {}: {:.2f}s ---'.format(idx, (time.time() - start_time)))

    def evaluate_idx(self, idx, test_loader):
        'Basic evaluate loop, NOT the hierarchical evaluation'
        print('Evaluating model id: {}'.format(idx))
        if torch.cuda.is_available():
            self.resnets[idx] = self.resnets[idx].cuda()
        self.resnets[idx].eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for _, (features, labels) in enumerate(test_loader):
                labels = self.change_labels(labels, idx)
                features, labels = Variable(features), Variable(labels)
                if torch.cuda.is_available():
                    features, labels = features.cuda(), labels.cuda()
                output = self.resnets[idx](features)
                loss += F.cross_entropy(output, labels, reduction='mean').data.item()
                # max(1) means along columms, [1] means get the indices list, not the actual values
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            loss /= len(test_loader)
            acc = float(100. * correct) / float(len(test_loader) * self.batch_size)
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f})%'.format(
                loss, correct, len(test_loader) * self.batch_size, acc))
        self.resnets[idx].cpu()
        return loss, acc

    def train(self):
        'Train the entire thingy'
        mean_accuracy = 0
        self.best_acc = []
        for epoch in range(self.epochs):
            epoch += 1
            loss, accuracy = [], []
            for idx in range(len(self.resnets)):
                self.train_idx(idx, epoch, self.train_loader)
            for idx in range(len(self.resnets)):
                ls, acc = self.evaluate_idx(idx, self.test_loader)
                loss.append(ls)
                accuracy.append(acc)
            # Epoch report
            print('\nEpoch {} completed!\nAverage validation loss: {:.4f}\tAverage accuracy: {:.2f}\n'.format(
                epoch, np.mean(loss), np.mean(accuracy)))
            # Update variables
            if mean_accuracy < np.mean(accuracy):
                mean_accuracy = np.mean(accuracy)
                self.best_acc = accuracy

    def recursive_evaluation(self, indices, data, labels, change, origin):
        """
        :param idx: index of the model we are evaluating
        :param data: features
        :param labels: labels
        :param change: numpy array of INDICES that need to be re-evaluated
        :param origin: recurse on this, basically the old predictions
        :return: pred, the predictions
        """
        # Else, do this stuff and recurse
        output = self.resnets[indices[0]](data)
        output = output.data.max(1, keepdim=True)[1].numpy().cpu()
        new_change = []
        for i in change:
            if output[i] == 1:
                origin[i] = indices[0]
            else:
                new_change.append(i)
        if new_change is not [] and len(indices) > 1:
            return self.recursive_evaluation(indices[1:], data, labels, new_change, origin)
        else:
            return origin


    def hierarchical_eval(self, data_loader):
        # First, we get the ordering of models
        self.best_acc = np.array(self.best_acc)
        self.indices = np.argsort(self.best_acc)
        correct = 0.
        total = 0.
        for data, labels in data_loader:
            dt, lb = Variable(data), Variable(labels)
            if torch.cuda.is_available():
                dt, lb = dt.cuda(), lb.cuda()
            pred = self.recursive_evaluation(self.indices, data, labels,
                                         np.array(list(range(len(labels)))),
                                         np.zeros(labels.shape))
            pred = torch.from_numpy(pred)
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            total += len(labels)
        acc = 100. * correct / total
        print('Hierarchical accuracy: {:.2f}'.format(acc))
        return acc
