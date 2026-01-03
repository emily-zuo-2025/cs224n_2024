import torch

x = torch.randn(3, requires_grad=True)
print (x)
y = x + 2
print (y)
z = y * y * 2
# z = z.mean() # make z as a scalar, otherwise the backward() won't work
print (z)
v = torch.tensor([1.0, 0.2, 0.001], dtype=torch.float32)
z.backward(v)  # dz/dx
print (x.grad)

# methods to not track gradient history
# 1. x.requires_grad_(False)
# 2. x.detach()
# 3. with torch.no_grad():
x.requires_grad_(False)
print (x)
y = x.detach()
print (y)
with torch.no_grad():
    y = x + 2
    print (y)
    
    
# important!! gradients can be accumulated
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    output = (weights * 3).sum()
    output.backward()
    print (weights.grad)
    weights.grad.zero_()
    
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()