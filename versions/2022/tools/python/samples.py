
import torch

class GNet(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super(GNet, self).__init__()
        self.mean = torch.Tensor([mean])
        self.std = torch.Tensor([std])
        
    def forward(self, x):
        return x*torch.normal(mean=self.mean, std=self.std)

x = GNet()
print(x(3))

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old."


x = Person("John", 30)
print(x)
