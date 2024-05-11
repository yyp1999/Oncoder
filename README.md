# Oncoder: deciphering tumor fractions using circulating cell-free DNA methylation for cancer detection
Oncoder takes cell-free DNA (cfDNA) from human blood as input and predicts tumor components in the blood using deconvolution. In the training stage, Oncoder learns a reference methylation profile from simulated data through supervised learning and applies it to predict tumor components.

![Overview](https://github.com/yyp1999/Oncoder/blob/main/Oncoder.png)


## Installation
This code doesn't require a special installation process if Python and necessary libraries are already installed. Download the Oncoder.py file and import in your Python script:
```python
import Oncoder
from Oncoder import Autoencoder
```

## Usage
See the deconvolution.ipynb notebook.

