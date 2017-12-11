import torch
import numpy as np

######################## Operations ############################################
x = torch.rand(5,3)
y = torch.rand(5,3)
# Addition: Syntax 0
print x+y

# Addition: Syntax 1
print torch.add(x,y)

# Addition: Syntax 2
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print result

# Addition: Syntax 3
y.add_(x)
print y
"""
Note:
Any operation that mutates a tensor in-place is post-fixed with an '_'
For example: x.copy_(y), x.t_(), will change x.

* torch can only be added to torch (e.g. torch + numpy = error)
"""


######################## Torch tensor to Numpy array ###########################
"""
The torch Tensor and numpy array will share their underlying memory locations,
and changing one will change the other.
"""
a = torch.ones(5)
print a

b = a.numpy()
print b

# a.add_(1) changes value of b as well as itself. However a=a+1 only changes itself
a.add_(1)
print b
print a

######################## Numpy array to Torch tensor ###########################


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)

print a
print b

"""
Note:

All the Tensors on the CPU except a CharTensor support converting to NumPy
and back.
"""

######################## Cuda Tenosrs ##########################################

# Tensors can be moved onto GPU using the .cuda function
if torch.cuda.is_available():
  x1 = torch.rand(5,3).cuda()
  y1 = torch.rand(5,3).cuda()
  print "Hello World!"
  print x+y

