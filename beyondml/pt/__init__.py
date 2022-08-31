"""
## PyTorch compatibility for building MANN models

The `beyondml.pt` subpackage contains layers and utilities for creating and pruning models using [PyTorch](https://pytorch.org).
The package contains two subpackages, the `beyondml.pt.layers` package, and the `beyondml.pt.utils` package.

Within the `layers` package, there is current functionality for the the following layers:
- `beyondml.pt.layers.Conv2D`
- `beyondml.pt.layers.Dense`
- `beyondml.pt.layers.FilterLayer`
- `beyondml.pt.layers.MaskedConv2D`
- `beyondml.pt.layers.MaskedDense`
- `beyondml.pt.layers.MultiConv2D`
- `beyondml.pt.layers.MultiDense`
- `beyondml.pt.layers.MultiMaskedConv2D`
- `beyondml.pt.layers.MultiMaskedDense`
- `beyondml.pt.layers.SelectorLayer`
- `beyondml.pt.layers.SparseConv2D`
- `beyondml.pt.layers.SparseDense`
- `beyondml.pt.layers.SparseMultiConv2D`
- `beyondml.pt.layers.SparseMultiDense`

Within the `beyondml.pt.utils` package, there is currently only one function, the `prune_model` function. Because of
the openness of developing with PyTorch in comparison to TensorFlow, there is far less functionality that
can be supplied directly via BeyondML. Instead, for converting models from training to inference, the user
is left to devise the best way to do so by building his or her own classes.

### Best Practices for Pruning
In order to use the `utils.prune_model` function, the model itself must have a `.layers` property. This property
is used to determine which layers can be pruned. **Only layers which support pruning and which are included in the
`.layers` property are pruned,** meaning the user can determine which exact layers in the model he or she wants
pruned. Alternatively, the user can create their own pruning function or method on the class itself and prune that way,
utilizing each of the `.prune()` methods of the layers provided.
"""

import beyondml.pt.layers
import beyondml.pt.utils
