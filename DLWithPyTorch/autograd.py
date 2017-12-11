import torch
from torch.autograd import Variable
"""
The autograd package provides automatic differentiation for all operations on Tensors.
It is a define-by-run framework, which means that your backprop is defined by
how your code is run, and that every single iteration can be different.
"""

######################## Variable ##############################################

"""
autograd.Variable.

This wraps a Tensor, and supports nearly all of operations defined on it.
Once you finish your computation you can call .backward() and have all the gradients
computed automatically.

You can acess the raw tensor through the .data attribute, while the gradient
w.r.t this variable is accumulated into .grad.

Variable and Function are interconnected and build up an acyclic graph,
that encodes a complete history of computation.

Each variable has a .grad_fn attribute references a Function that has created
the Variable. Except for Variables created by the user -  their grad_fn is None

If Variable holds a one element data, you don't need to specify any arguments
to backward(), however if it has more elements, you need to specify a "grad_ouput"
argument that is a tensor of matching shape.
"""

# Create a variable
x = Variable(torch.ones(2,2), requires_grad=True)
print x

# Do an operation of variable, y becomes a variable too.
y = x + 2
print y
print y.grad_fn

# More operations
z = y * y * 3
out = z.mean()

print z.grad_fn
print out.grad_fn

######################## Gradients #############################################

out.backward()
print x.grad
print y.grad

"""
Only gives gradients of variables created by users
print y.grad gives us "None"
"""

########## Gradients of individual elements of varialbe and Learning rate ######

"""
As mentioned above, when the final output is not a scalar then you have to put
a Tensor which is the same shape as variable. Also each element of that tensor
is a learning rate.
"""

x = Variable(torch.ones(3), requires_grad=True)
y = x*2
y.backward(torch.Tensor([0.1, 1.0, 0.0001]))
print x.grad

