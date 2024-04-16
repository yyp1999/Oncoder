# Oncoder: deciphering tumor fractions using circulating cell-free DNA methylation for cancer detection
Oncoder takes cell-free DNA (cfDNA) from human blood as input and predicts tumor components in the blood using deconvolution. In the training stage, Oncoder learns a reference methylation profile from simulated data through supervised learning and applies it to predict tumor components.

## Installation
This code doesn't require a special installation process if Python and necessary libraries are already installed. Download the Oncoder.py file and import in your Python script:
'''python
import Oncoder
from Oncoder import Autoencoder
'''

## Usage
see the deconvolution.ipynb notebook.

## File instruction
The 'data' directory contains three type of datasets: reference_data, HCC_data and normal_data.

  _reference_data_
  A matrix used to generate simulation patients data and the prior beta distribution of the reference methylation atlas.  Index should be CpG and columns should be sample type('GSE40279','LIHC')

  _HCC_data and normal_data_
  Chip-based methylation data from GSE129374 and GSE41169. Data missing specific probes have been excluded.
  
