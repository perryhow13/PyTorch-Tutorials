e"""
Computational graphs and autograd are a very powerful paradigm for defining complex
operators and automatically taking derivatives; however for large neural networks
raw autograd can be a bit too low-level.

The nn package serves higher-level abstractions over raw computational graphs that
are useful for building neural networks.

The nn package defines a set of Moduels, which are roughly equivalent to neural network
layers.

Each module receives input Variables and computes ouput Variables, but may also
hold internal state such as Varaibles containing learnable parameters.

The nn package also defines a set of useful loss functions that are commonly used
when training neural networks.
"""

import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)


# Use the nn pacakge to define our model as a sequence of layers.
# nn.Sequential is a Module whihc contains other Modules, and applies them in sequence
# to produce its output.

# Each Linear function hold internal Variables for its weight and bias.
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out))


loss_fn = torch.nn.MSELoss(size_average=False)
learning_rate = 1e-4

for i in range(500):

  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  print(i, loss.data[0])

  # Zero the gradients before running the backward pass.
  # Note we only call zero_grad() when the graph is consist of modules of nn package.
  model.zero_grad()

  loss.backward()

  for param in model.parameters():
    param.data -= learning_rate * param.grad.data

