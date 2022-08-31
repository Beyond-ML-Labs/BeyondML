"""
## TensorFlow compatibility for building MANN models.

The `beyondml.tflow` package contains two subpackages, `beyondml.tflow.layers` and `beyondml.tflow.utils`, which contain
the functionality to create and train MANN layers within TensorFlow. For individuals who are
familiar with the former name of this package, `mann`, backwards compatibility can be achieved
(assuming only TensorFlow support is needed), by replacing the following line of code:

>>> import mann

with the following line:

>>> import beyondml.tflow as mann

in all existing scripts.

Within the `layers` package, there is current functionality for the the following layers:
- `beyondml.tflow.layers.FilterLayer`
- `beyondml.tflow.layers.MaskedConv2D`
- `beyondml.tflow.layers.MaskedDense`
- `beyondml.tflow.layers.MultiConv2D`
- `beyondml.tflow.layers.MultiDense`
- `beyondml.tflow.layers.MultiMaskedConv2D`
- `beyondml.tflow.layers.MultiMaskedDense`
- `beyondml.tflow.layers.MultiMaxPool2D`
- `beyondml.tflow.layers.SelectorLayer`
- `beyondml.tflow.layers.SumLayer`
- `beyondml.tflow.layers.SparseDense`
- `beyondml.tflow.layers.SparseConv`
- `beyondml.tflow.layers.SparseMultiDense`
- `beyondml.tflow.layers.SparseMultiConv`

**Note that with any of the sparse layers (such as the `SparseDense` layer), any model which
utilizes these layers will not be loadable using the traditional `load_model` functions available
in TensorFlow. Instead, the model should be saved using either joblib or pickle.**

Within the `utils` package, there are the current functions and classes:
- `ActiveSparsification`
- `build_transformer_block`
- `build_token_position_embedding_block`
- `get_custom_objects`
- `mask_model`
- `remove_layer_masks`
- `add_layer_masks`
- `quantize_model`
- `get_task_masking_gradients`
- `mask_task_weights`
- `train_model_iteratively`
"""

import beyondml.tflow.layers
import beyondml.tflow.utils
