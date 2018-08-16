## load data

import random
import numpy as np

# random.seed(42)

def make_data():
    x1 = random.randint(0, 1)
    x2 = random.randint(0, 1)
    yy = 0 if (x1 == x2) else 1

    x1 = 2. * (x1 - 0.5)
    x2 = 2. * (x2 - 0.5)
    yy = 2. * (yy - 0.5)

    # add noise
    x1 += 0.1 * random.random()
    x2 += 0.1 * random.random()
    yy += 0.1 * random.random()

    return [x1, x2], yy

batch_size = 10
def make_batch():
    data = [make_data() for ii in range(batch_size)]
    labels = [label for xx, label in data]
    data = [xx for xx, label in data]
    return np.array(data, dtype='float32'), np.array(labels, dtype='float32')

print(make_batch())
print(make_batch())
print(make_batch())

train_data = [make_batch() for ii in range(500)]
test_data = [make_batch() for ii in range(50)]

## model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# torch.manual_seed(42)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        # first layer with 2 inputs and 2 outputs
        self.dense1 = nn.Linear(2, 2)
        # second layer with 2 inputs and 1 output
        self.dense2 = nn.Linear(2, 1)

        # 2 * 2 matrix
        print(self.dense1.weight)
        # 1 * 2 matrix
        print(self.dense1.bias)
        # 1 * 2 matrix
        print(self.dense2.weight)
        # 1 * 1 matrix for the next layer node input.
        print(self.dense2.bias)

        # self.dense1.weight.data.uniform_(-1.0, 1.0)
        # self.dense1.bias.data.uniform_(-1.0, 1.0)
        # self.dense2.weight.data.uniform_(-1.0, 1.0)
        # self.dense2.bias.data.uniform_(-1.0, 1.0)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return torch.squeeze(x)

model = NN()

## optimizer = stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

## train and test functions
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = Variable(torch.from_numpy(data)), Variable(torch.from_numpy(target))
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} {}\tLoss: {:.4f}'.format(epoch, batch_idx * len(data), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_data:
        data, target = Variable(torch.from_numpy(data), volatile=True), Variable(torch.from_numpy(target))
        output = model(data)
        test_loss += F.mse_loss(output, target)
        correct += (np.around(output.data.numpy()) == np.around(target.data.numpy())).sum()

    test_loss /= len(test_data)
    test_loss = test_loss.data[0]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, batch_size * len(test_data), 100. * correct / (batch_size * len(test_data))) )

## run

nepochs = 10
for epoch in range(1, nepochs + 1):
    train(epoch)
    test()