import torch

import numpy as np
import h5py

import matplotlib
import matplotlib.pyplot as plt

import glob

plt.rcParams.update({'mathtext.fontset':'cm'})

class SubgridDataset(torch.utils.data.Dataset):
  def __init__(self, device, path, samples, size, x, y, established=0):
    self.device = device
    self.path = path
    self.samples = samples - established
    self.size = size
    self.established = established
    self.x = x
    self.y = y
    self.inputs = torch.zeros(self.samples, len(x), size, size, size, dtype=torch.float32)
    self.labels = torch.zeros(self.samples, len(y), size, size, size, dtype=torch.float32)

    for i, f in enumerate(x):
      for j, c in enumerate(sorted(glob.glob(path + f + '_*'))):
        if j < established:
          # dynamic not established
          continue
        elif j >= samples:
          break

        # transform
        p = j - established
        file = h5py.File(c, 'r')
        dset = file[f]
        self.inputs[p][i] = torch.from_numpy(dset[()])

    for i, f in enumerate(y):
      for j, c in enumerate(sorted(glob.glob(path + f + '_*'))):
        if j < established:
          # dynamic not established
          continue
        elif j >= samples:
          break

        # transform
        p = j - established
        #to_pytorch(self.labels[p][i], f, c)
        file = h5py.File(c, 'r')
        dset = file[f]
        self.labels[p][i] = torch.from_numpy(dset[()])

  def plot(self, ylabels, plot_samples=5, slice=5):
    features_inputs = self.inputs.shape[1]
    features_labels = self.labels.shape[1]
    features = features_inputs + features_labels
    fig, ax = plt.subplots(
      nrows=features,
      ncols=plot_samples + 1,
      figsize=(plot_samples * 2.5, features * 2.5),
      constrained_layout=True,
      gridspec_kw={"width_ratios": np.append(np.repeat(1, plot_samples), 0.05)}
    )
    fig.suptitle(r'Cut in dataset throughout the simulation of $\mathcal{S}$')

    max_input, _ = self.inputs[:,:,slice].flatten(start_dim=2).max(2)[0].max(0)
    min_input, _ = self.inputs[:,:,slice].flatten(start_dim=2).min(2)[0].min(0)
    max_label, _ = self.labels[:,:,slice].flatten(start_dim=2).max(2)[0].max(0)
    min_label, _ = self.labels[:,:,slice].flatten(start_dim=2).min(2)[0].min(0)
    for j in range(plot_samples):
      sample = int(
        (j) * 
        ((self.samples - 1) / 
        (plot_samples - 1) ))

      off = len(self.x)
      for i, f in enumerate(self.x):
        c = ax[i, j].contourf(self.inputs[sample][i][slice], 100, cmap='bwr', vmin=min_input[i], vmax=max_input[i])
        if j < 1:
          # add label
          ax[i, j].set_ylabel(ylabels[i], fontsize=20.0)
          fig.colorbar(c, cax=ax[i, plot_samples])
      for i, f in enumerate(self.y):
        l = ax[i + off, j].contourf(self.labels[sample][i][slice], 100, cmap='bwr', vmin=min_label[0], vmax=max_label[0])
        if j < 1:
          # add label
          ax[i + off, j].set_ylabel(ylabels[i + off], fontsize=20.0)
          fig.colorbar(l, cax=ax[i + off, plot_samples])
      ax[i + off, j].set_xlabel(r'$\mathcal{S}_{' + str((self.established + sample + 1) * 500) + r'}$', fontsize=20.0)
    #fig.savefig('dataset.png')

  def __len__(self):
    return self.samples

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    inputs = self.inputs[idx]
    labels = self.labels[idx]

    return (inputs.to(self.device), labels.to(self.device))
