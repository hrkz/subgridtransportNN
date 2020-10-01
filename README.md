# Subgrid Transport Neural Network

This repository contains code for the paper [Embedding physical knowledge into neural networks for sub-grid scale modeling in filtered turbulence](arxiv:?) (2020).

<img src="https://github.com/hrkz/SubgridTransportNN/data/fig/subgrid_turbulence.png" width="500">

## Dataset

The dataset is available [link] and should be extracted in ``data/`` by default. It contains DNS data filtered at different resolutions, even if this paper only deal with a filter size equal to 8.

## Usage

Three notebooks can be found in ``notebook/`` that shows how to load the data, train a model and evaluate pretrained version with the different metrics presented in the paper. 
The source of the SGTNN model can be found otherwise in ``src/``.

## Citing

If you find this code useful in your research, consider citing with
```
@article{}
```
