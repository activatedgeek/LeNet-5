from lenet import LeNet5
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Scale((32, 32)),
                       transforms.ToTensor(),
                   ]))
data_test = MNIST('./data/mnist',
                   train=False,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Scale((32, 32)),
                       transforms.ToTensor(),
                   ]))
data_train_loader = DataLoader(data_train, batch_size=1024, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

net = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)


def train(epoch):
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images), Variable(labels)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)
        print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data[0]))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = Variable(images), Variable(labels)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data[0], float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()

def main():
    for e in range(1, 16):
        train_and_test(e)


if __name__ == '__main__':
    main()
