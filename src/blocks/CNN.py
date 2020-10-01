import torch
import numpy as np

class PeriodicPadding3d(torch.nn.Module):
  def __init__(self, size):
    super(PeriodicPadding3d, self).__init__()
    self.size = size

  def forward(self, x):
    bc = np.pad(x.cpu().detach(), 
      ((0, 0), 
       (0, 0), 
       (self.size, self.size), 
       (self.size, self.size), 
       (self.size, self.size)), 'wrap'
    )
    return torch.from_numpy(bc).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

class ConvUnit(torch.nn.Module):
  def __init__(self, in_size, out_size, kernel=3):
    super(ConvUnit, self).__init__()

    self.x = in_size
    self.y = out_size
    self.k = kernel

    self.conv = torch.nn.Conv3d(in_size, out_size, kernel, stride=1, padding=0, bias=False)
    self.norm = torch.nn.BatchNorm3d(out_size)

    #torch.nn.init.zeros_(self.conv.bias)
    torch.nn.init.kaiming_normal_(self.conv.weight)

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    return torch.nn.functional.relu(x, inplace=True)

class UnitaryUnit(torch.nn.Module):
  def __init__(self, in_size):
    super(UnitaryUnit, self).__init__()

    self.x = in_size
    self.k = 1
    self.conv = torch.nn.Conv3d(in_size, 1, 1, stride=1, padding=0, bias=False)

  def forward(self, x):
    x = self.conv(x)
    return x

class BlockCNN(torch.nn.Module):
  def __init__(self, name, layers):
    super(BlockCNN, self).__init__()
    self.name = name
    self.layers = torch.nn.ModuleList(layers)

    # insert periodic padding based on convolution layers
    self.layers.insert(
      0,
      PeriodicPadding3d(
        size=int(sum([(layer.k - 1) / 2 for layer in layers]))
      )
    )

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
