# MWC-Net: Multiscale Wavelet-based Spatial-Spectral Compression Network for Hyperspectral Image
## Introduction
Hyperspectral images (HSIs) possess high-dimensional tensor structures that present significant reconstruction challenges 
under ultra-low compression ratios (CR) in artificial intelligence-driven remote sensing. Conventional compression 
methods are unable to effectively capture inherent spatial-spectral coherence and often neglect multiscale spectral 
absorption-reflection dependencies, which are critical for maintaining spectral fidelity. 
To overcome these shortcomings, we propose a Multiscale Wavelet-based Spatial-Spectral Compression Network (MWC-Net) 
for HSI reconstruction. Methodologically, MWC-Net integrates a three-dimensional (3D) spatial-spectral attention encoder, 
which via tri-branch attention to extract complete spatial-spectral coherence. Additionally, we develop a multiscale 
wavelet spatial-spectral decoder that restores scale-sensitive spectral features through multiscale super-resolution 
and enhances spatial-spectral resolution using wavelet decomposition.

## Requirements
* Ununtu 18.0 
* python 3.8 
* Pytorch 1.6 

## Datasets
Data address: https://aviris.jpl.nasa.gov/dataportal/.

After downloading, save it to the dataset folder.

## Training and Testing
The paper is currently under review. Once accepted, the test code and weight file will be released.
