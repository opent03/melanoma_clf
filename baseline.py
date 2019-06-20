import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision.models as torchmodels
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def load_split_train_test(datadir, valid_size=.2, batch_size=32, num_workers=6, size=(200,200)):
    normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transforms = transforms.Compose([transforms.Resize(size),
                                           transforms.ToTensor(),
                                           normalize,
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(size),
                                          transforms.ToTensor(),
                                          normalize,
                                      ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    test = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)
    return train, test

# Setup variables
batch_size = 128
epochs = 15
learning_rate = 1e-3
imgsize = (128, 128)
root = 'data/'
train_loader, test_loader = load_split_train_test(root, batch_size=batch_size, size=imgsize)

# Model
net = torchmodels.resnet50(pretrained=True)
ct = 0
for child in net.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
net.fc = nn.Linear(2048, 8)

optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    net = net.cuda()
    criterion = criterion.cuda()

# Init weights
def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
net.apply(init_weights)


def train(net, train_loader, optimizer, criterion, epoch, batch_size):
    'Train loop'
    net.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, target = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 20 == 0:
            batch_size = len(images)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(images), len(train_loader) * batch_size,
                100. * (batch_idx + 1) / len(train_loader), loss.item()))


def evaluate(model, data_loader, batch_size):
    'Evaluate loop'
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(data_loader):
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            output = model(data)

            loss += F.cross_entropy(output, target, size_average=False).data.item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()


        loss /= len(data_loader.dataset)
        acc = float(100. * correct) / float(len(data_loader)*batch_size)
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)'.format(
            loss, correct, len(data_loader)*batch_size, acc))
    return acc

# debug
'''
train_data = datasets.ImageFolder('data/')
sample_dest = 'data/0MEL/ISIC_00000002.jpg'
for img, label in train_loader:
    img = transforms.ToPILImage()(img[0])
    print(label)
    img = np.array(img)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    exit()
exit()
'''
for epoch in range(epochs):
    epoch += 1
    train(net, train_loader, optimizer, criterion, epoch, batch_size)
    #print('train accuracy: ')
    #tracc = evaluate(net, train_loader, batch_size)
    print('test_accuracy: ')
    teacc = evaluate(net, test_loader, batch_size)