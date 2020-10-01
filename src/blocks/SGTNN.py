import torch
import numpy as np

from blocks.CNN import PeriodicPadding3d, ConvUnit, UnitaryUnit

class StandardizeUnit(torch.nn.Module):
  def __init__(self):
    super(StandardizeUnit, self).__init__()
    
  def forward(self, x):
    y = x.clone()
   
    for i in range(0, y.shape[0]):
      for j in range(0, 3):
        y[i, j] = (y[i, j] - y[i, j].mean())
    return y

class LinearConvUnit(torch.nn.Module):
  def __init__(self, in_size, out_size, kernel=3):
    super(LinearConvUnit, self).__init__()

    self.x = in_size
    self.y = out_size
    self.k = kernel

    self.conv = torch.nn.Conv3d(in_size, out_size, kernel, stride=1, padding=0, bias=False)
    #torch.nn.init.kaiming_normal_(self.conv.weight)

  def forward(self, x):
    x = self.conv(x)
    return x

class SGTNN(torch.nn.Module):
  def __init__(self, name):
    super(SGTNN, self).__init__()
    self.name = name
    
    layers = [
        ConvUnit(in_size= 3, out_size=  8, kernel=3), 
        ConvUnit(in_size= 8, out_size= 16, kernel=3), 
        ConvUnit(in_size=16, out_size= 32, kernel=3), 
        ConvUnit(in_size=32, out_size= 64, kernel=3), 
        ConvUnit(in_size=64, out_size=128, kernel=3)
    ]
    
    subgrid = [StandardizeUnit(), PeriodicPadding3d(size=int(sum([(layer.k - 1) / 2 for layer in layers])))]
    subgrid.extend(layers)
    
    self.subgrid   = torch.nn.ModuleList(subgrid)
    self.transport = torch.nn.ModuleList([PeriodicPadding3d(size=3), LinearConvUnit(1, 128, 7)])
    
    self.merge = UnitaryUnit(in_size=128)
    
  def forward(self, x):
    t = x[:, 3]
    t = torch.unsqueeze(t, 1)
    for layer in self.transport:
        t = layer(t)
    y = x[:,:3]
    for layer in self.subgrid:
        y = layer(y)
        
    y = torch.mul(t, y)
    y = self.merge(y)
    return y
