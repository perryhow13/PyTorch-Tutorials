import torch
from torch.autograd import Variable

# Manually implementing the backward pass is not a big deal for
# for a small two-layer network, but can quickly get very hairy
# for large complex networks.

# Automatic differentiation to automate the computation of backward
# passes in neural networks.

# The autograd package in PyTorch provides exactly this functionality.
# When using autograd, the forward pass of your network will define
# a computational graph; nodes in the graph will be Tensors, and edges
# will be functions that produce output Tensors from input Tensors.
# Backpropagationg through this graph then allows you to easily compute gradients.

# We wrap our PyTorch Tensors in Variable objects; a Variable represents a node in a
# computational graph. If x is a Variable then x.data is a Tensor, and x.grad is
# another Variable holding the gradient of x with respect to some scalar value.

# PyTorch Variables have the same API as PyTorch Tensors:(almost)any operation
# that you can perform on a Tensor also works on Variables;
# The difference is that using Variables defines a computational graph, allowing
# you to automatically compute gradients

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6
for i in range(500):
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = 0.5*(y_pred - y).pow(2).sum()
  print(i, loss.data[0])
  loss.backward()

  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Tensor.zero_(): Fills the tensor with zeros
  w1.grad.data.zero_()
  w2.grad.data.zero_()


