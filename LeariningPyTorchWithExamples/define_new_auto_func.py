# So far, just like we did in autograd.py, we have implemented
# the computational graph and made each operation to do
# forward pass and backward pass. (I wrote this)

# Each primitive autograd operator is really two functions
# that operate on Tensors, forward and backward function.
# We can easily define our own autograd operator by defining
# a subclass of torch.autograd.Function. We can then use
# our new autograd operator by constructiing an instance
# and calling it like a function, passing Variables containing
# input data.
import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and impelmenting the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):

    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):

    # , only applies to a vector of the shape (1,n) to make the vector (n,1)
    # Note! , does not apply to the vector of the shape (n,1) or any other shape.
    input, = self.saved_tensors

    # Tensor.clone() returns a copy of the tensor.
    # Note it is not Variable.clone()
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input

N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6

for t in range(500):
  # Construct an instance of our MyReLU class to use in our network
  relu = MyReLU()

  # Forward pass: compute predicted y using operations on Variables; we compute
  # ReLU using our custom autograd operation.
  y_pred = relu(x.mm(w1)).mm(w2)

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.data[0])

  # Use autograd to compute the backward pass.
  loss.backward()

  # Update weights using gradient descent
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients after updating weights
  w1.grad.data.zero_()
  w2.grad.data.zero_()












