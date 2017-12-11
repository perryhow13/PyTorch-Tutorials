"""
This code only involves small modification of nn.py.
Up to this point, we have been updating learnable parameters manually by mutating
the '.data' of each Variable holding learnable parameters.

When we use SGD, it is simple to implement manually.

But when we use more sophisticiated optimizers, such as AdaGrad, RMSProp, Adam, etc.
It is more promising to use optim pacakge.

Optim abstracts the idea of an optimization algorithm and provides implementations of
commonly used optimization algorithms

"""

import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.ReLU(),
                            torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(500):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)
  print(i, loss.data[0])

  optimizer.zero_grad()

  # Backpropagate to compute gradients
  loss.backward()

  # Update the learnable parameters using defined optimization algorithm.
  optimizer.step()
