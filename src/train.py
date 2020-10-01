import torch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import IPython.display as idisplay

import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mean squared error
def mse(y, y_hat):
  e = (y - y_hat)**2
  return e.mean()

# integral dissipation
def dis(divtau, theta):
  d = divtau * theta
  return d.mean()

def loss(x, y, y_hat):
  # coupled mse / dissipation loss
  e = mse(y, y_hat)
  return e

def train(net, dataset, opti, rate, stat):
  net.train()
  cost = 0.0
  for step, batch in enumerate(dataset):
    opti.zero_grad()
    data, labs = batch

    pred = net(data)
    grad = loss(data, pred, labs)
    
    grad.backward()
    opti.step()
    rate.step()

    cost  += grad.item()

  cost /= len(dataset)
  stat.append(cost)

def valid(net, dataset, stat):
  net.eval()

  cost = 0.0
  with torch.no_grad():
    for step, batch in enumerate(dataset):
      data, labs = batch

      pred = net(data)
      grad = loss(data, pred, labs)
      cost += grad.item()

    cost /= len(dataset)
  stat.append(cost)

def loop(net, model_path, train_loader, valid_loader, opti, rate, epochs=1000):
    
  path = model_path + '/' + net.name

  if not os.path.isdir(path):
    # create directory for model save and losses
    os.mkdir(path)
    
  loss_fig = plt.figure(figsize=(15,5))
  loss_axs = loss_fig.add_subplot(1, 1, 1)

  train_loss = []
  valid_loss = []

  for epoch in range(1, epochs + 1):
    train(net, train_loader, opti, rate, train_loss)
    valid(net, valid_loader, valid_loss)
    
    if epoch % 10 == 0:
      print('Current loss (epoch {}) = {}'.format(epoch, train_loss[-1]), flush=True)
      loss_axs.set_xlim(0, epoch)
      loss_axs.cla()
      loss_axs.semilogy(train_loss, label='Training loss')
      loss_axs.semilogy(valid_loss, label='Validation loss')
      loss_axs.legend()
      loss_axs.grid(True, linestyle='--')
      loss_fig.savefig(path + '/loss.png')
      idisplay.display(loss_fig)
      idisplay.clear_output(wait=True)
    if epoch % 100 == 0:
      np.savetxt(path + '/losses.csv', np.column_stack((train_loss, valid_loss)), delimiter=",", fmt='%s')
      torch.save(net.state_dict(), path + '/weights.pyt')
  
  print('Finished training, with last progress = {}'.format(train_loss[-1] - train_loss[-2]))
