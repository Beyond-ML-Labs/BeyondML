"""
MANN - Multitask Artificial Neural Network package

This package contains custom utilities and layers to use to build Multitask Artificial Neural Network (MANN) models
in conjunction with TensorFlow and Keras.
"""

__version__ = '0.3.0'
__dev__ = True

import mann.layers
import mann.utils
import warnings

warnings.warn(
    'MANN is being deprecated and will be replaced in the future by the beyondml package. Version 0.3.0 will be the FINAL version of MANN. Please be advised.',
    DeprecationWarning,
    stacklevel=2
)
