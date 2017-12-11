
"""
Neural networks can be constructed using the torch.nn package. nn depends on
autograd to define models and differentiate them.

An nn.Module contains layers, and a method forward(input) that returns the output.

"""

######################## Define the network ####################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# nn.Module is a parent class and Net is going to be its child class
class Net(nn.Module):
  def __init__(self):
    # super(Net, self).__init__() for using cooperative multiple inheritance.
    # without super, Net.__init__(self), limits your ability to use multiple inheritance.
    super(Net, self).__init__()

    # 1 input image channel, 6 output channels, 5x5 square convolution kernel
    self.conv1 = nn.Conv2d(1, 6, 5)
    # 6 input image channel, 16 output channels, 5x5 square convolution filter
    self.conv2 = nn.Conv2d(6, 16, 5)
    #an affine operation: y = Wx +b

    self.fc1 = nn.Linear(16*5*5,120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84,10)

  def forward(self, x):

    # Max pooling over a (2,2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # If the size is a square you can only specify a single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
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

"""
Note: We just have defined the forward function, and the backward function is
      automatically defined for you using autograd.
"""

# The learnable parameters of a model are returned by net.parameters()
net = Net()
params = list(net.parameters())
print(len(params))
for i in range(len(params)):
  print(params[i].size())

# These are kernels
print params[0]
# These are bias
print params[1]

# The input to the forward is an autograd.Variable, and so is the output.
input = Variable(torch.randn(1,1,32,32))
out = net(input)

######################## Loss Function #########################################

# Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1,10))

"""
Note:

torch.nn only supports mini-batches. The entire torch.nn pacakge only supports
inputs that are a mini-batch of samples, and not a single sample.

For example, nn.Conv2d will take in a 4D Tensor of
nSamples x nChannels X Height x Width

If you have a single sample, just use input.unsqueeze(0) to add a fake batch
dimension.
"""

# A loss function takes the (output, target) pair of inputs, and computes a
# value that estimated how far away the output is from the target.

output = net(input)
target = Variable(torch.arange(1,11))
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

######################## Back Propagation ######################################
# Before backpropgate the error all we have to do is to loss.backward(). You
# need to clear the existing gradients, else gradients will be accumulated to
# existing gradients
net.zero_grad()
loss.backward()

######################## Update the weights ####################################
# Use the simplest update rule, SGD: weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
  f.data.sub_(f.grad.data * learning_rate)


# ALTERNATIVE WAY (i.e. Simpler way)

# creat your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()




