"""
Sometimes you will want to specificy models that are more complex than a
sequence of exisitng Modules;

You can define your own Modules by subclassing nn.Module and defining a forward
which recieves input Variables and produces output Varaibles using other modules
or other autograd operations on Variables.

So I think it is possible to define my own autograd operations and use it when
defining the layer.

autograd operations are functions that work on Variables.
But nn modules are layers which include various autograd operations.
So the custom nn is defining new layer using other layers and autograd operators.
Note layers are consist of autograd operators. So customizing nn Module is essentially
just defining new layer using bunch of autograd operators.
"""

import torch
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)
  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred

N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)
model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=le-4)

for i in range(500):
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  print(i, loss.data[0])

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()