import numpy as np
import torch

x = torch.empty(2, 2, 3)
x = torch.rand(2, 2)
x = torch.zeros(2, 2)
x = torch.ones(2, 2)
x = torch.ones(2, 2, dtype=torch.int)
x = torch.tensor([2.5, 0.1])
# print (x.size())
# print(x)
# Basic operations of tensors
x = torch.rand(2, 2)
y = torch.rand(2, 2)
# print (x)
# print (y)
z = x + y
# do the same thing as above
z = torch.add(x, y)
#print (z)
# in-place operation will modify the value
y.add_(x)
#print (y)
# muliplication
z = x * y
z = torch.mul(x, y)
y.mul_(x)

x = torch.rand(5, 3)
# print (x)
# print ()
# print (x[:, 0])
# print()
# print (x[1, :])
# print (x[1, 1].item())

# Reshape dimensations by using view
x = torch.rand(4, 4)
# print (x)
# print ()
y = x.view(8, 2)
y = x.view(-1, 8)
# print (y.size())

# conversion between numpy array and tensors
#a = torch.ones(5)
# print (a)
#b = a.numpy()
# print (b)
# print (type(b))
# a and b point to the same memory location. Be careful!!
#a.add_(1)
# print (a)
# print (b)

a = np.ones(5)
print (a)
b = torch.from_numpy(a)
print (b)

