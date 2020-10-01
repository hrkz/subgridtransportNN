import torch

import numpy as np

import scipy
import scipy.stats
import scipy.spatial.distance

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mse(x, y):
  res = (x - y)**2
  return res.mean()

def diss(divtau, theta):
  res = divtau * theta
  return res.mean()

def evaluate_mse(alpha, beta, dataset, models):
  tests_batch = 4
  tests_loader = torch.utils.data.DataLoader(dataset, batch_size=tests_batch, shuffle=False)
  
  for x in models:
    x.eval()
    
  preds = [None] * len(models)
    
  dataset.inputs[:,  3] *= alpha  
  dataset.inputs[:, :3] += beta
    
  with torch.no_grad():
    for step, batch in enumerate(tests_loader):
      data, labs = batch
      if step == 0:
        for i, x in enumerate(models):
          preds[i] = x(data)
      else:
        for i, x in enumerate(models):
          preds[i] = torch.cat((preds[i], x(data)), axis=0)
    
  dataset.inputs[:, :3] -= beta
  dataset.inputs[:,  3] /= alpha
    
  data, labs = dataset[:]

  for i, x in enumerate(models):
    preds[i] = preds[i] / alpha

  mse_eval = {}
  for i, x in enumerate(models):
    mse_eval[x.name] = mse(preds[i][:, 0], labs[:, 0])
  return mse_eval

def evaluate(dataset, models):
  tests_batch = 4
  tests_loader = torch.utils.data.DataLoader(dataset, batch_size=tests_batch, shuffle=False)
  
  for x in models:
    x.eval()
    
  preds = [None] * len(models)

  with torch.no_grad():
    for step, batch in enumerate(tests_loader):
      data, labs = batch
      if step == 0:
        for i, x in enumerate(models):
          preds[i] = x(data)
      else:
        for i, x in enumerate(models):
          preds[i] = torch.cat((preds[i], x(data)), axis=0)
    
  data, labs = dataset[:]

  mse_eval = {}
  mse_eval['smag'] = mse(labs[:, 1], labs[:, 0])
  mse_eval['rg']   = mse(labs[:, 2], labs[:, 0])
  for i, x in enumerate(models):
    mse_eval[x.name] = mse(preds[i][:, 0], labs[:, 0])

  diss_truth = diss(labs[:, 0], data[:, 3])

  diss_eval = {}
  diss_eval['smag'] = (diss(labs[:, 1], data[:, 3]) - diss_truth)
  diss_eval['rg']   = (diss(labs[:, 2], data[:, 3]) - diss_truth)
  for i, x in enumerate(models):
    diss_eval[x.name] = (diss(preds[i][:, 0], data[:, 3]) - diss_truth)

  # switch to numpy for convenience
  data = data.detach().cpu().numpy()
  labs = labs.detach().cpu().numpy()

  for i in range(0, len(models)):
    preds[i] = preds[i].detach().cpu().numpy()

  cc_eval = {}
  cc_eval['smag'] = scipy.stats.pearsonr(labs[:, 1].flatten(), labs[:, 0].flatten())[0]
  cc_eval['rg']   = scipy.stats.pearsonr(labs[:, 2].flatten(), labs[:, 0].flatten())[0]
  for i, x in enumerate(models):
    cc_eval[x.name] = scipy.stats.pearsonr(preds[i][:, 0].flatten(), labs[:, 0].flatten())[0]

  # distr
  rng = [min(labs.min(), min(x.min() for x in preds)), max(labs.max(), max(x.max() for x in preds))]  
  m_h = np.histogram(200 * ((labs[:, 0].flatten() - rng[0]) / (rng[1] - rng[0])) - 100, bins=range(-100, 100), density=True)[0]
  m_s = np.histogram(200 * ((labs[:, 1].flatten() - rng[0]) / (rng[1] - rng[0])) - 100, bins=range(-100, 100), density=True)[0]
  m_r = np.histogram(200 * ((labs[:, 2].flatten() - rng[0]) / (rng[1] - rng[0])) - 100, bins=range(-100, 100), density=True)[0]
    
  hists = [None] * len(models)
  for i, x in enumerate(models):
    hists[i] = np.histogram(200 * ((preds[i][:, 0].flatten() - rng[0]) / (rng[1] - rng[0])) - 100, bins=range(-100, 100), density=True)[0]
                      
  m_s[m_s  == 0] = 1e-12
  m_r[m_r  == 0] = 1e-12
                                     
  for h in hists:
    h[h == 0] = 1e-12
  
  div_eval = {}    
  div_eval['smag'] = scipy.spatial.distance.jensenshannon(m_h, m_s)
  div_eval['rg']   = scipy.spatial.distance.jensenshannon(m_h, m_r)
  for i, x in enumerate(models):
    div_eval[x.name] = scipy.spatial.distance.jensenshannon(m_h, hists[i])
    
  m_s[m_s  == 1e-12] = 0
  m_r[m_r  == 1e-12] = 0

  for h in hists:
    h[h == 1e-12] = 0

  ks_eval = {}
  ks_eval['smag'] = scipy.stats.ks_2samp(labs[:, 1].flatten(), labs[:, 0].flatten())[0]
  ks_eval['rg']   = scipy.stats.ks_2samp(labs[:, 2].flatten(), labs[:, 0].flatten())[0]
  for i, x in enumerate(models):
    ks_eval[x.name] = scipy.stats.ks_2samp(preds[i][:, 0].flatten(), labs[:, 0].flatten())[0]

  print(
    '''\t\t MSE (L2) \t\t Dissipation error (I) \t Cross-correlation (P)\n
    \t Smag  \t {} \t {} \t {}\n
    \t Rg    \t {} \t {} \t {}\n'''.format(
      mse_eval['smag'], diss_eval['smag'], cc_eval['smag'],
      mse_eval['rg'],   diss_eval['rg'],   cc_eval['rg'],
  ))

  for x in models:
    print('\t{} \t {} \t {} \t {}\n'.format(x.name, mse_eval[x.name], diss_eval[x.name], cc_eval[x.name]))
    
  print(
    '''\t\t JS distance (J) \t KS test (K)\n
    \t Smag  \t {} \t {}\n
    \t Rg    \t {} \t {}\n'''.format(
      div_eval['smag'], ks_eval['smag'],
      div_eval['rg'],   ks_eval['rg']
  ))

  for x in models:
    print('\t{} \t {} \t {}\n'.format(x.name, div_eval[x.name], ks_eval[x.name]))

  # plots
  samples = [0, int(dataset.samples * 1 / 4), int(dataset.samples / 2), int(dataset.samples * 3 / 4), dataset.samples - 1]
  sliceid = int(dataset.size / 2)

  # div
  div_fig, div_axes = plt.subplots(
    nrows=3 + len(models),
    ncols=5 + 1,
    figsize=(12.5,(3 + len(models))*2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(1, 5), 0.05)}
  )

  div_fig.suptitle(r'$\nabla \cdot \mathbf{s}$', fontsize=20)

  rng = [
      labs[samples[:], 0, sliceid].min(), 
      labs[samples[:], 0, sliceid].max()
  ]
    
  div_axes[0,0].contourf(labs[samples[0], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[0,1].contourf(labs[samples[1], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[0,2].contourf(labs[samples[2], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[0,3].contourf(labs[samples[3], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  c1 = div_axes[0,4].contourf(labs[samples[4], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[1,0].contourf(labs[samples[0], 1, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[1,1].contourf(labs[samples[1], 1, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[1,2].contourf(labs[samples[2], 1, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[1,3].contourf(labs[samples[3], 1, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[1,4].contourf(labs[samples[4], 1, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[2,0].contourf(labs[samples[0], 2, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[2,1].contourf(labs[samples[1], 2, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[2,2].contourf(labs[samples[2], 2, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[2,3].contourf(labs[samples[3], 2, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  div_axes[2,4].contourf(labs[samples[4], 2, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
  for i, x in enumerate(models):
    div_axes[3+i,0].contourf(preds[i][samples[0], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
    div_axes[3+i,1].contourf(preds[i][samples[1], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
    div_axes[3+i,2].contourf(preds[i][samples[2], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
    div_axes[3+i,3].contourf(preds[i][samples[3], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
    div_axes[3+i,4].contourf(preds[i][samples[4], 0, sliceid], 100, cmap='bwr', vmin=rng[0], vmax=rng[1])
    
  div_fig.colorbar(c1, cax=div_axes[0,5])
  div_fig.colorbar(c1, cax=div_axes[1,5])
  div_fig.colorbar(c1, cax=div_axes[2,5])
  for i, x in enumerate(models):
    div_fig.colorbar(c1, cax=div_axes[3+i,5])
    
  div_axes[0,0].set_ylabel(r'$\mathcal{M}$', fontsize=15)
  div_axes[1,0].set_ylabel(r'$\mathcal{M}_{\mathrm{DynSmag}}$', fontsize=15)
  div_axes[2,0].set_ylabel(r'$\mathcal{M}_{\mathrm{DynRG}}$', fontsize=15)
  for i, x in enumerate(models):
    div_axes[3+i,0].set_ylabel(r'$\mathcal{M}_{\mathrm{' + x.name + '}}$', fontsize=15)

  len_div = 3 + len(models) -1
  div_axes[len_div,0].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[0] + 1) * 500) + r'}$', fontsize=15)
  div_axes[len_div,1].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[1] + 1) * 500) + r'}$', fontsize=15)
  div_axes[len_div,2].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[2] + 1) * 500) + r'}$', fontsize=15)
  div_axes[len_div,3].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[3] + 1) * 500) + r'}$', fontsize=15)
  div_axes[len_div,4].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[4] + 1) * 500) + r'}$', fontsize=15)

  # div error
  ediv_fig, ediv_axes = plt.subplots(
    nrows=2 + len(models),
    ncols=5 + 1,
    figsize=(12.5,(2 + len(models))*2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(1, 5), 0.05)}
  )
    
  ediv_fig.suptitle(r'$|\nabla \cdot \mathbf{s} - \widehat{\nabla \cdot \mathbf{s}}|$', fontsize=20)

  ediv_smag = np.abs(labs[:, 0] - labs[:, 1])
  ediv_rg   = np.abs(labs[:, 0] - labs[:, 2])

  errors = [None] * len(models)
  for i, x in enumerate(models):
    errors[i] = np.abs(labs[:, 0] - preds[i][:, 0])
    
  rng = [
      min(
        ediv_smag[samples, sliceid].min(), 
        ediv_rg  [samples, sliceid].min(), 
        min(x[samples, sliceid].min() for x in errors)
      ),
      max(
        ediv_smag[samples, sliceid].max(), 
        ediv_rg  [samples, sliceid].max(),
        max(x[samples, sliceid].max() for x in errors)
      )
  ]

  ediv_axes[0,0].contourf(ediv_smag[samples[0], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[0,1].contourf(ediv_smag[samples[1], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[0,2].contourf(ediv_smag[samples[2], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[0,3].contourf(ediv_smag[samples[3], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[0,4].contourf(ediv_smag[samples[4], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[1,0].contourf(ediv_rg  [samples[0], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[1,1].contourf(ediv_rg  [samples[1], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[1,2].contourf(ediv_rg  [samples[2], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  ediv_axes[1,3].contourf(ediv_rg  [samples[3], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  c1 = ediv_axes[1,4].contourf(ediv_rg[samples[4], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
  for i, x in enumerate(models):
    ediv_axes[2+i,0].contourf(errors[i][samples[0], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
    ediv_axes[2+i,1].contourf(errors[i][samples[1], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
    ediv_axes[2+i,2].contourf(errors[i][samples[2], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
    ediv_axes[2+i,3].contourf(errors[i][samples[3], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])
    ediv_axes[2+i,4].contourf(errors[i][samples[4], sliceid], 100, cmap='coolwarm', vmin=rng[0], vmax=rng[1])

  ediv_fig.colorbar(c1, cax=ediv_axes[0,5])
  ediv_fig.colorbar(c1, cax=ediv_axes[1,5])
  for i, x in enumerate(models):
    ediv_fig.colorbar(c1, cax=ediv_axes[2+i,5])
    
  ediv_axes[0,0].set_ylabel(r'$\mathcal{M}_{\mathrm{DynSmag}}$', fontsize=15)
  ediv_axes[1,0].set_ylabel(r'$\mathcal{M}_{\mathrm{DynRG}}$', fontsize=15)
  for i, x in enumerate(models):
    ediv_axes[2+i,0].set_ylabel(r'$\mathcal{M}_{\mathrm{' + x.name + '}}$', fontsize=15)

  len_ediv = 2 + len(models) -1
  ediv_axes[len_ediv,0].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[0] + 1) * 500) + r'}$', fontsize=15)
  ediv_axes[len_ediv,1].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[1] + 1) * 500) + r'}$', fontsize=15)
  ediv_axes[len_ediv,2].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[2] + 1) * 500) + r'}$', fontsize=15)
  ediv_axes[len_ediv,3].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[3] + 1) * 500) + r'}$', fontsize=15)
  ediv_axes[len_ediv,4].set_xlabel(r'$\mathcal{S}_{' + str((dataset.established + samples[4] + 1) * 500) + r'}$', fontsize=15)

  pdf_div_fig, pdf_div_axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5,5),
    constrained_layout=True
  )
    
  m_h = np.histogram(labs[:, 0].flatten(), bins=200, density=True)
  m_s = np.histogram(labs[:, 1].flatten(), bins=200, density=True)
  m_r = np.histogram(labs[:, 2].flatten(), bins=200, density=True)

  pdf = [None] * len(models)
  for i, x in enumerate(models):
    pdf[i] = np.histogram(preds[i][:, 0].flatten(), bins=200, density=True)
    
  pdf_div_axes.plot(m_h[1][1:], m_h[0], label=r'$\mathcal{M}_{\mathrm{DNS}}$')
  pdf_div_axes.plot(m_s[1][1:], m_s[0], label=r'$\mathcal{M}_{\mathrm{DynSmag}}$')
  pdf_div_axes.plot(m_r[1][1:], m_r[0], label=r'$\mathcal{M}_{\mathrm{DynRG}}$')
  for i, x in enumerate(models):
    pdf_div_axes.plot(pdf[i][1][1:], pdf[i][0], label=r'$\mathcal{M}_{\mathrm{' + x.name + '}}$')
    
  pdf_div_axes.set_yscale('log', nonpositive='clip')
  pdf_div_axes.set_ylim(1e-6, 10)
  pdf_div_axes.legend(fontsize=15)
  pdf_div_axes.grid(True, linestyle='--')
  pdf_div_axes.set_ylabel(r'$pdf$', fontsize=20)
  pdf_div_axes.set_xlabel(r'$\nabla \cdot \mathbf{s}$', fontsize=20)
    
  cdf_div_fig, cdf_div_axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5,5),
    constrained_layout=True
  )

  cdf_div_axes.plot(m_h[1][1:], np.cumsum(m_h[0]) / m_h[0].sum(), label=r'$\mathcal{M}_{\mathrm{DNS}}$')
  cdf_div_axes.plot(m_s[1][1:], np.cumsum(m_s[0]) / m_s[0].sum(), label=r'$\mathcal{M}_{\mathrm{DynSmag}}$')
  cdf_div_axes.plot(m_r[1][1:], np.cumsum(m_r[0]) / m_r[0].sum(), label=r'$\mathcal{M}_{\mathrm{DynRG}}$')
  for i, x in enumerate(models):
    cdf_div_axes.plot(pdf[i][1][1:], np.cumsum(pdf[i][0]) / pdf[i][0].sum(), label=r'$\mathcal{M}_{\mathrm{' + x.name + '}}$')

  cdf_div_axes.set_yscale('log', nonpositive='clip')
  cdf_div_axes.set_ylim(1e-6, 5)
  cdf_div_axes.legend(fontsize=15)
  cdf_div_axes.grid(True, linestyle='--')
  cdf_div_axes.set_ylabel(r'$cdf$', fontsize=20)
  cdf_div_axes.set_xlabel(r'$\nabla \cdot \mathbf{s}$', fontsize=20)

  qq_stride = int(dataset.samples / 2)

  qq_div_fig, qq_div_axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5,5),
    constrained_layout=True
  )

  _, qq_x    = scipy.stats.probplot(labs[:, 0].flatten(), fit=False)
    
  m_hv, m_hx, m_hy = np.histogram2d(qq_x, scipy.stats.probplot(labs[:, 1].flatten(), fit=False)[1], bins=400)
  m_sv, m_sx, m_sy = np.histogram2d(qq_x, scipy.stats.probplot(labs[:, 2].flatten(), fit=False)[1], bins=400)
    
  dens = [None] * len(models)
  for i, x in enumerate(models):
    dens[i] = np.histogram2d(qq_x, scipy.stats.probplot(preds[i][:, 0].flatten(), fit=False)[1], bins=400)
    
  ax_hv = np.where(m_hv != 0)
  ax_sv = np.where(m_sv != 0)
    
  downsp = [None] * len(models)
  for i, x in enumerate(models):
    downsp[i] = np.where(dens[i][0] != 0)

  qq_div_axes.scatter(m_hx[ax_hv[0]], m_hy[ax_hv[1]], 10.0, label=r'$\mathcal{M}_{\mathrm{DynSmag}}$')
  qq_div_axes.scatter(m_sx[ax_sv[0]], m_sy[ax_sv[1]], 10.0, label=r'$\mathcal{M}_{\mathrm{DynRG}}$')
  for i, x in enumerate(models):
    qq_div_axes.scatter(dens[i][1][downsp[i][0]], dens[i][2][downsp[i][1]], 10.0, label=r'$\mathcal{M}_{\mathrm{' + x.name + '}}$')
    
  qq_div_axes.plot([np.min(labs[:, 0]), np.max(labs[:, 0])], [np.min(labs[:, 0]), np.max(labs[:, 0])], 'k--')
  qq_div_axes.legend(fontsize=15)
  qq_div_axes.grid(True, linestyle='--')
  qq_div_axes.set_xlabel(r'$cdf^{-1}_{\mathcal{M}_{\mathrm{DNS}}} \, \, \nabla \cdot \mathbf{s}$', fontsize=20)
  qq_div_axes.set_ylabel(r'$cdf^{-1}_{\mathcal{M}} \, \, \nabla \cdot \mathbf{s}$', fontsize=20)
