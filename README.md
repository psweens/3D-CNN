# 3-dimensional Segmentation using a 3D Convolution Neural Network

This Python package utilised a 3-dimensional convolution neural network (CNN) to perform segmentation of 3D images using [Keras](https://keras.io/). The 3D CNN is based on the [U-Net architecture](https://arxiv.org/abs/1505.04597) but extended for [volumetric delineation](https://arxiv.org/abs/1606.06650) with 3D spatial convolutions. 

Hyperparameters can be tuned using [Talos](https://github.com/autonomio/talos) which integrates with [Keras](https://keras.io/). This automates and performs a quantitative evaluation of the CNN during training.

To-date, this script has been used to train a 3D CNN which predicts a tumour region-of-interest from 3D raster-scan optoacoustic mesoscopy (RSOM) images, found [here](https://github.com/psweens/Predict-RSOM-ROI/blob/main/README.md).

## References 
To reference this repository please use the below citation.

> [Quantification of vascular networks in photoacoustic mesoscopy](https://www.sciencedirect.com/science/article/pii/S221359792200026X)<br>
> Emma L. Brown, Thierry L. Lefebvre, Paul W. Sweeney et al.

## Prerequisites
The 3D CNN was trained using [Keras](https://keras.io/) using the following package versions:
* Python 3.6.
* Keras 2.3.1.
* Tensorflow-GPU 1.14.0.

A package list for a Python environment has been provided and can be installed using the method described below.

## Installation
The ROI package is compatible with Python3, and has been tested on Ubuntu 18.04 LTS. 
Other distributions of Linux, macOS, Windows should work as well.

To install the package from source, download zip file on GitHub page or run the following in a terminal:
```bash
git clone https://github.com/psweens/Predict-RSOM-ROI.git
```

The required Python packages can be found [here](https://github.com/psweens/3D-CNN/blob/main/REQUIREMENTS.txt). The package list can be installed, for example, using creating a Conda environment by running:
```bash
conda create --name <env> --file REQUIREMENTS.txt
```
