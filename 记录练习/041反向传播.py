import torch
a = torch.tensor([1.0])
a.requires_grad = True
print(a)
print(a.data)
print(a.type)  #  a 类型是temsor
print(a.data.type)   #  a.data 的类型是tensor
print(a.grad)
print(type(a.grad))
