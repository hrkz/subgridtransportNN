# Subgrid Transport Neural Network

This repository contains code for the paper [Physical invariance in neural networks for subgrid-scale scalar flux modeling](https://arxiv.org/abs/2010.04663) (2020).

<img src="data/fig/subgrid_turbulence.png?raw=true" width="500">

## Dataset

The dataset is available [here](https://zenodo.org/record/4067946) and should be extracted in ``data/`` by default. It contains DNS data filtered at different resolutions, even if this paper only deal with a filter size equal to 8.

## Usage

Three notebooks can be found in ``notebook/`` that shows how to load the data, train a model and evaluate pretrained version with the different metrics presented in the paper. 
The source of the SGTNN model can be found otherwise in ``src/``.

## Citing

If you find this code useful in your research, consider citing with
```
@article{frezat2020physical,
  title={Physical invariance in neural networks for subgrid-scale scalar flux modeling},
  author={Frezat, Hugo and Balarac, Guillaume and Sommer, Julien Le and Fablet, Ronan and Lguensat, Redouane},
  journal={arXiv preprint arXiv:2010.04663},
  year={2020}
}
```
