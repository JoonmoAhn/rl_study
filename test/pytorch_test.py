import torch
x = torch.rand(5,2)
y = torch.rand(3,3)
print(x)
print(y)

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
