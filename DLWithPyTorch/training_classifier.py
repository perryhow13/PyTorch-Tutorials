######################## What about data? ######################################

# Specifically for image data, there is a package called torch vision, which
# has data loaders for well known datasets such as Imagenet, CIFAR10, MNIST.etc.

import torch
import torchvision
import torchvision.transforms as transforms

######################## Training an image classifier ##########################
# Let us try to load CIFAR10 using torchvision.
# The output of torchvision datasets are PILImage images of range [0,1].
# We transform them to Tensors of normalized range [-1,1]

"""
transforms.ToTensor(): Convert a PIL Image or numpy.ndarray to tensor
transforms.Normalize(mean,std): Given mean: (M0, ..., Mn)
                                Given std: (S0,..., Sn) for n channels.
transforms.Compose(transforms): Given transforms: a list of Transform objects -
                                                  list of transforms to compose.
                                So each element of the list can be chained together
                                using Compose.
"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./MNIST', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,shuffle=False, num_workers=2)

"""
import matplotlib.pyplot as plt
import numpy as numpy

def imshow(img):
  img = img/2 + 0.5 #unnormalize
  npimg = img.numpy
  plt.imshow(np.transpose(npimg, (1,2,0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = datiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

######################## Define a Convolution Neural Network ###################

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(32, 64, 5)
    self.fc1 = nn.Linear(64*4*4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64*4*4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    # all dimensions excpet the batch dimension
    size = x.size()[1:]
    num_features = 1

    for s in size:
      num_features *= s

    return num_features

net = Net()
#net.cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

######################## Train the Convolution Neural Network ##################

# loop over the dataset multiple times
for epoch in range(10):
  running_loss = 0.0
  # Here enumerate(..., 0) means, index starts from 0.
  # If enumerate(..., 1) then index starts from 1
  for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.data[0]

    if i  == 1199:
      print('[%d, %5d] mean loss: %.3f' % (epoch + 1, i + 1, running_loss / 1200))
      running_loss = 0.0

print('Finished Training')

######################## Test the Convolution Neural Network ###################

# Make iterator, so there are 60000/batch_size (i.e. number of batches) in the iterator.
# So when we call data_iter.next, the next batch comes out in order.
# data_iter = iter(testloader).

# 1 is dimension, since this is a batch, dimension 0 is a batch index
# 1 is the probabilities for each class of the batch.
# torch.max() outputs(max, max_indices) thats why _ is done.

"""dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)
"""

correct = 0.0
total = 0.0

for data in testloader:
  images, labels = data
  outputs = net(Variable(images))
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %.4f %%' % (
    100.0 * correct / total))


