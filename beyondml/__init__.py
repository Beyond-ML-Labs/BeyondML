"""
BeyondML (formerly MANN) is a Python package which enables creating sparse multitask artificial neural networks (MANNs)
compatible with [TensorFlow](https://tensorflow.org) and [PyTorch](https://pytorch.org). This package
contains custom layers and utilities to facilitate the training and optimization of models using the
Reduction of Sub-Network Neuroplasticity (RSN2) training procedure developed by [AI Squared, Inc](https://squared.ai).

### Installation

This package is available through [PyPi](https://pypi.org) and can be installed via the following command:

```bash
pip install beyondml
```

### Capabilities

There are two major subpackages within the BeyondML package, the `beyondml.tflow` and the `beyondml.pt` packages.
The `beyondml.tflow` package contains functionality for building multitask models using TensorFlow, and the
`beyondml.pt` package contains functionality for building multitask models using PyTorch.
"""

__version__ = '0.1.3'
__dev__ = True

import beyondml.tflow
import beyondml.pt
