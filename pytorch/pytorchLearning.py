import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # imshow(torchvision.utils.make_grid(images))
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16*5*5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# net = Net()
# print(net)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
# loss_f = nn.CrossEntropyLoss()
#
# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         output = net(inputs)
#         optimizer.zero_grad()
#         loss = loss_f(output, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(loss.item())
#             print(loss)
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#
#
# print('Finished Training')
# PATH = './cifar_net.path'
# torch.save(net.state_dict(), PATH)
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16*6*6, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *=s
#         return num_features
#
#
#
# net = Net()
# print(net)
#
# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)
#






# x = torch.unsqueeze(torch.linspace(-1, 1, 100),dim=1)
# y = x.pow(2) + 0.2*torch.randn(x.size())
# class Net(nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
#
# net = Net(1, 10, 1)
# print(net)
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# loss_f = torch.nn.MSELoss()
#
# for t in range(1000):
#     pre = net(x)
#     loss = loss_f(pre, y)
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# x  = np.mat('0, 0;'
#             '0 1;'
#             '1 0;'
#             '1 1')
# print(x)
# x = torch.tensor(x).float()
# print(x)
# y = np.mat('1;'
#            '0;'
#            '0;'
#            '1')
# y = torch.tensor(y).float()
#
# myNet = nn.Sequential(
#     nn.Linear(2, 10),
#     nn.ReLU(),
#     nn.Linear(10, 1),
#     nn.Sigmoid()
# )
#
# print(myNet)
#
# optimer = torch.optim.SGD(myNet.parameters(), lr=0.05)
# loss_func = nn.MSELoss()
#
# for epoch in range(10000):
#     out = myNet(x)
#     loss = loss_func(out, y)
#     print(loss)
#     optimer.zero_grad()
#     loss.backward()
#     optimer.step()
#
# print(myNet(x).data)
# x = torch.ones(3, requires_grad=True)
# y = x*2
# while y.data.norm()< 1000:
#     y = y*2
#
#
# print(y)
# v = torch.tensor([0.1, 2.0, 0.01], dtype=torch.float)
# y.backward(v)
# print(x.grad)


# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# print(y.grad_fn)
# z = y * y * 3
# out = z.sum()
#
# print(z, out)
# out.backward()
# print(x.grad)
# x = torch.tensor([1.0, 2.0], requires_grad=True)
# print(x)
# y = x*x + 2
# print(y)
# y = y.sum()
# y.backward()
# print(x.grad)
# a = torch.randn(2, 2)
# a = ((a*3)/(a-1))
# print(a.requires_grad)
# a.requires_grad_(True)
# b = (a*a).sum()
# print(b)

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)
# np.add(a, 1, out=a)
# print(a)
# print(b)
# x = torch.ones(5, 3, dtype=torch.float)
# y = torch.randn(5, 3, dtype=torch.float)
# a = torch.ones(5)
# b = a.numpy()
# print(a)
# print(b)
# a.add_(1)
# print(a)
# print(b)
# print(y.view(15))
# print(y.view(5, -1).size())
# print(y)
# print(y[1:2, 1:3])
# print(x)
# print(y)
# print(x + y)
# result = torch.empty(5, 3)
# torch.add(x, y, out= result)
# print(result)
# y.add_(x)
# print(y)
# x = torch.tensor([3.5, 5])
# print(x)
# x = torch.empty([5, 3])
# print(x)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)