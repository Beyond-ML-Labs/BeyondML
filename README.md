# MANN

MANN, which stands for Multitask Artificial Neural Networks, is a Python package which enables creating sparse multitask models compatible with [TensorFlow](https://tensorflow.org). This package contains custom layers and utilities to facilitate the training and optimization of models using the Reduction of Sub-Network Neuroplasticity (RSN2) training procedure developed by [AI Squared, Inc](https://squared.ai).

## Installation

This package is available through [PyPi](https://pypi.org) and can be installed via the following command:

```bash
pip install mann
```

To install the current version directly from [GitHub](https://github.com) without cloning, run the following command:

```bash
pip install git+https://github.com/AISquaredInc/mann.git
```

Alternatively, you can install the package by cloning the repository from [GitHub](https://github.com) using the following commands:

```bash
# clone the repository and cd into it
git clone https://github.com/AISquaredInc/mann
cd mann

# install the package
pip install .
```

### Mac M1 Users

For those with a Mac with the M1 processor, this package can be installed, but the standard version of TensorFlow is not compatible with the M1 SOC. In order to install a compatible version of TensorFlow, please install the [Miniforge](https://github.com/conda-forge/miniforge) conda environment, which utilizes the conda-forge channel only. Once you are using Miniforge, using conda to install TensorFlow in that environment should install the correct version. After installing TensorFlow, the command `pip install mann` will install the MANN package.

## Capabilities

The MANN package includes two subpackages, the `mann.utils` package and the `mann.layers` package. As the name implies, the `mann.utils` package includes utilities which assist in model training. The `mann.layers` package includes custom Keras-compatible layers which can be used to train sparse multitask models.

### Utils

The `mann.utils` subpackage contains helper functions for performing training and conversion of models using masking layers.

In addition to the functions just mentioned, there is also an `ActiveSparsification` callback object which enables active sparsification during training rather than solely one-shot sparsification. Note that this callback currently only supports simultaneous training. We are working to support iterative training with this callback as well.

1. `mask_model`
    - The `mask_model` function is central to the RSN2 training procedure and enables masking/pruning a model so a large percentage of the weights are inactive.
    - Inputs to the `mask_model` function are a TensorFlow model, a percentile in integer form, a method - either one of 'gradients' or 'magnitude', input data, and target data.
2. `get_custom_objects`
    - The `get_custom_objects` function takes no parameters and returns a dictionary of all custom objects required to load a model trained using this package.
3. `remove_layer_masks`
    - The `remove_layer_masks` function takes a trained model with masked layers and converts it to a model without masking layers.
4. `add_layer_masks`
    - The `add_layer_masks` function takes an existing model that has non-MANN layers and converts it so that all layers which have an analog in the MANN package. This enables pretrained models to be converted and sparsified.
5. `quantize_model`
    - The `quantize_model` function takes in a model and a datatype to quantize the model to.
6. `build_transformer_block`
    - The `build_transformer_block` function can be used to build a block in a transformer architecture.
7. `build_token_position_embedding`
    - The `build_token_position_embedding` function can be used to build a token and position embedding block for use in a transformer architecture model.

### Layers

The `mann.layers` subpackage contains custom Keras-compatible layers which can be used to train sparse multitask models. The layers contained in this package are as follows:

1. `MaskedDense`
    - This layer is nearly identical to the Keras Dense layer, but it supports masking and pruning to reduce the number of active weights.
2. `MaskedConv2D`
    - This layer is nearly identical to the Keras Conv2D layer, but it supports masking and pruning to reduce the number of active weights.
3. `MultiMaskedDense`
    - This layer supports isolating pathways within the network and dedicating them for individual tasks and performing fully-connected operations on the input data.
4. `MultiMaskedConv2D`
    - This layer supports isolating pathways within the network and dedicating them for individual tasks and performing convolutional operations on the input data.
5. `MultiDense`
    - This layer supports multitask inference using a fully-connected architecture and is not designed for training. Once a model is trained with the `MultiMaskedDense` layer, that layer can be converted into this layer for inference by using the `mann.utils.remove_layer_masks` function.
6. `MultiConv2D`
    - This layer supports multitask inference using a convolutional architecture and is not designed for training. Once a model is trained with the `MultiMaskedConv2D` layer, that layer can be converted to this layer for inference by using the `mann.utils.remove_layer_masks` function.
7. `SelectorLayer`
    - This layer selects which of the multiple inputs fed into it is returned as a result. This layer is designed to be used specifically with multitask layers.
8. `SumLayer`
    - This layer returns the element-wise sum of all of the inputs.
9. `FilterLayer`
    - This layer can be turned on or off, and indicates whether the single input passed to it should be output or if all zeros should be returned.
10. `MultiMaxPool2D`
    - This layer implements Max Pool operations on multitask inputs.

## Additional Documentation and Training Materials

Additional documentation and training materials will be added to the [BeyondML Documentation Website](https://docs.squared.ai/projects/beyondml-labs) as we continue to develop this project and its capabilities.

## Feature Roadmap

- Transformers
    - We currently support basic functionality for the Transformer architecture with this package. We are working to expand our capabilities with this architecture to further support greater use cases with it, including pruning.
    
## Changes

Below are a list of additional features, bug fixes, and other changes made for each version.

### Version 0.2.2
- Small documentation changes
- Added `quantize_model` function
- Added `build_transformer_block` and `build_token_position_embedding_block` functions for transformer functionality
- Removed unnecessary imports breaking imports in minimal environments
