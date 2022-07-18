"""
TensorFlow compatibility for building MANN models.

The `beyondml.tflow` package contains two subpackages, `layers` and `utils`, which contain
the functionality to create and train MANN layers within TensorFlow. For individuals who are
familiar with the former name of this package, `mann`, backwards compatibility can be achieved
(assuming only TensorFlow support is needed), by replacing the following line of code:

>>> import mann

with the following line:

>>> import beyondml.tflow as mann

in all existing scripts.
"""

import beyondml.tflow.layers
import beyondml.tflow.utils
