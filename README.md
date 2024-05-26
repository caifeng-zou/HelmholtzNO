# HNO: Deep Neural Helmholtz Operators

## Introduction
This repository provides code and (part of) data for the paper [Deep Neural Helmholtz Operators for 3D Elastic Wave Propagation and Inversion](https://arxiv.org/pdf/2311.09608).

## File Description
- **code**: 
    - **util**: Classes and functions involved
    - HNO_3D_train.ipynb: Workflow for training a 3D HNO
    - HNO_GNO_3D_train.ipynb: Workflow for training a 3D GNO-embedded HNO 
    - HNO_GNO_3D_fwi.ipynb: Workflow for full-waveform inversion
    - HNO_3D_test_overthrust.ipynb: Generalization test with overthrust models
- **data**: Data for use with the code
- **data_generation**: Code for generating training data with [Salvus](https://mondaic.com/docs/0.12.15/getting_started)
- **model**: Normalizers for data processing. For 3D HNO models please see below.

## Dependencies
```
environment.yml
```

## Pre-trained 3D Models
The 3D HNO models can be reproduced with code and data in this repository. Pre-trained models are also available [here](https://drive.google.com/drive/folders/1T10Bv0wj09u5vUY_WqZdWdmqJHPtUCha?usp=drive_link).

## Example Prediction
![video](animation_toverthrust.gif)

## Contact
We welcome any comments or questions regarding this work. If you find it helpful, please cite:
```
Zou, C., Azizzadenesheli, K., Ross, Z. E., & Clayton, R. W. (2023). Deep Neural Helmholtz Operators for 3D elastic wave propagation and inversion. arXiv preprint arXiv:2311.09608.
```

Caifeng Zou\
czou@caltech.edu

