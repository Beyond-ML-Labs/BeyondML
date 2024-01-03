.. BeyondML documentation master file, created by
   sphinx-quickstart on Fri Jan  6 12:23:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/BeyondML_horizontal-color.png
   :align: center
   :width: 400

|

Welcome to BeyondML's documentation!
====================================

BeyondML is a Python package which enables creating sparse multitask artificial neural networks (MANNs)
compatible with `TensorFlow <https://tensorflow.org>`_ and `PyTorch <https://pytorch.org>`_.
This package contains custom layers and utilities to facilitate the training and optimization of models
using the Reduction of Sub-Network Neuroplasticity (RSN2) training procedure developed by `AI Squared, Inc <https://squared.ai>`_.

:download:`View this Documentation in PDF Format <./_build/latex/beyondml.pdf>`

Installation
************

This package is available through `Pypi <https://pypi.org>`_ and can be installed by running the following command:

.. code-block::

   pip install beyondml

Alternatively, the latest version of the software can be installed directly from GitHub using the following command:

.. code-block::

   pip install git+https://github.com/beyond-ml-labs/beyondml

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   modules

Changelog
*********

- Version 0.1.0
   - Refactored existing MANN repository to rename to BeyondML
- Version 0.1.1
   - Added the `SparseDense`, `SparseConv`, `SparseMultiDense`, and `SparseMultiConv` layers to 
      `beyondml.tflow.layers`, giving users the functionality to utilize sparse tensors during 
      inference
- Version 0.1.2
   - Added the `MaskedMultiHeadAttention`, `MaskedTransformerEncoderLayer`, and `MaskedTransformerDecoderLayer` layers to `beyondml.pt.layers` to add pruning to the transformer architecture
   - Added `MaskedConv3D`, `MultiMaskedConv3D`, `MultiConv3D`, `MultiMaxPool3D`, `SparseConv3D`, and `SparseMultiConv3D` layers to `beyondml.tflow.layers`
   - Added `MaskedConv3D`, `MultiMaskedConv3D`, `MultiConv3D`, `MultiMaxPool3D`, `SparseConv3D`, `SparseMultiConv3D`, and `MultiMaxPool2D` layers to `beyondml.pt.layers`
- Version 0.1.3
   - Added `beyondml.pt` compatibility with more native PyTorch functionality for using models on different devices and datatypes
   - Added `train_model` function to `beyondml.tflow.utils`
   - Added `MultitaskNormalization` layer to `beyondml.tflow.layers` and `beyondml.pt.layers`
- Version 0.1.4
   - Updated documentation to use Sphinx
- Version 0.1.5
   - Updated requirements to use newer version of TensorFlow
   - Fixed errors with changes to types of `input_shape` in TensorFlow Keras layers
   - Fixed errors resulting from model/configuration changes with TensorFlow
- Version 0.1.6
   - Fixed issues with converting between masked and unmasked models in TensorFlow
- Version 0.1.7
   - Updated Pytorch implementation of Transformer-based architectures