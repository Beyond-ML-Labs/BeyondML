![](https://github.com/Beyond-ML-Labs/artwork/blob/main/horizontal/color/BeyondML_horizontal-color.png)

# BeyondML (formerly MANN)

[![Documentation](https://badgen.net/badge/icon/Documentation?icon=chrome&label)](https://beyond-ml-labs.github.io/BeyondML/beyondml.html)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6190/badge)](https://bestpractices.coreinfrastructure.org/projects/6190)
[![PyPI version](https://badge.fury.io/py/beyondml.svg)](https://badge.fury.io/py/beyondml)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

BeyondML is a Python package which enables creating sparse multitask artificial neural networks (MANNs) compatible with [TensorFlow](https://tensorflow.org) and [PyTorch](https://pytorch.org). This package contains custom layers and utilities to facilitate the training and optimization of models using the Reduction of Sub-Network Neuroplasticity (RSN2) training procedure developed by [AI Squared, Inc](https://squared.ai).

## Installation

This package is available through [PyPi](https://pypi.org) and can be installed via the following command:

```bash
pip install beyondml
```

To install the current version directly from [GitHub](https://github.com) without cloning, run the following command:

```bash
pip install git+https://github.com/Beyond-ML-Labs/BeyondML.git
```

Alternatively, you can install the package by cloning the repository from [GitHub](https://github.com) using the following commands:

```bash
# clone the repository and cd into it
git clone https://github.com/Beyond-ML-Labs/BeyondML
cd BeyondML

# install the package
pip install .
```

### Mac M1 Users

For those with a Mac with the M1 processor, this package can be installed, but the standard version of TensorFlow is not compatible with the M1 SOC. In order to install a compatible version of TensorFlow, please install the [Miniforge](https://github.com/conda-forge/miniforge) conda environment, which utilizes the conda-forge channel only. Once you are using Miniforge, using conda to install TensorFlow in that environment should install the correct version. After installing TensorFlow, the command `pip install beyondml` will install the BeyondML package.

## Contributing

For those who are interested in contributing to this project, we first thank you for your interest! Please refer to the CONTRIBUTING.md file in this repository for information about best practices for how to contribute.

### Vulnerability reporting

In the event you notice a vulnerability within this project, please open a [GitHub Issue](https://github.com/Beyond-ML-Labs/BeyondML/issues) detailing the vulnerability to report it. In the event you would like to keep the report private, please email <mann@squared.ai>.

## Capabilities

To view current capabilities within the BeyondML package, we welcome you to check the [BeyondML documentation](https://beyond-ml-labs.github.io/BeyondML/beyondml.html).

## Feature Roadmap
Lists of features slated for this project will be added here.

## Changes

Below are a list of additional features, bug fixes, and other changes made for each version.

### MANN

The below version numbers and logged changes refer to the MANN package.

#### Version 0.2.2
- Small documentation changes
- Added `quantize_model` function
- Added `build_transformer_block` and `build_token_position_embedding_block` functions for transformer functionality
- Removed unnecessary imports breaking imports in minimal environments

#### Version 0.2.3
- Per-task pruning
  - Functionality for this feature is implemented, but usage is expected to be incomplete. Note that task gradients have to be passed retrieved and passed to the function directly (helper function available), and that the model has to initially be compiled using a compatible loss function (recommended 'mse') to identify gradients.
  - It has been found that this functionality is currently only supported for models with the following layers:
    - MaskedConv2D
    - MaskedDense
    - MultiMaskedDense
  - Note also that this functionality does not support cases where layers of an individual model are other TensorFlow models, but supporting this functionality is on the roadmap.
- Iterative training using per-task pruning
  - Functionality for this feature is implemented, but there are known bugs when trying to apply this methodology to models with the `MultiMaskedConv2D` layer present

#### Version 0.3.0
- Support for PyTorch layers
- Support for additional custom objects in the `quantize_model` function
- Added tests to the package functionality
- Added auto-generated documentation

### BeyondML

The below version numbers and changes refer to the BeyondML package

#### Version 0.1.0
- Refactored existing MANN repository to rename to BeyondML

#### Version 0.1.1
- Added the `SparseDense`, `SparseConv`, `SparseMultiDense`, and `SparseMultiConv` layers to 
  `beyondml.tflow.layers`, giving users the functionality to utilize sparse tensors during 
  inference
