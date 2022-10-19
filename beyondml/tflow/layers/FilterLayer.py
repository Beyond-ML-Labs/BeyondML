import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class FilterLayer(Layer):
    """
    Layer which filters inputs based on status of `on` or `off`

    Example:

    >>> # Create a model with just a FilterLayer
    >>> input_layer = tf.keras.layers.Input(10)
    >>> filter_layer = mann.layers.FilterLayer()(input_layer)
    >>> model = tf.keras.models.Model(input_layer, filter_layer)
    >>> model.compile()
    >>> # Call the model with the layer turned on
    >>> data = np.arange(10).reshape((1, 10))
    >>> model.predict(data)
    array([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]], dtype=float32)
    >>> # Turn off the FilterLayer and call it again
    >>> model.layers[-1].turn_off()
    >>> # Model must be recompiled after turning the layer on or off
    >>> model.compile()
    >>> model.predict(data)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)

    """

    def __init__(
            self,
            is_on=True,
            **kwargs
    ):
        super(FilterLayer, self).__init__(**kwargs)
        self.is_on = is_on

    def call(self, inputs):
        """
        This is where the layer's logic lives and is called upon inputs

        Parameters
        ----------
        inputs : TensorFlow Tensor or Tensor-like
            The inputs to the layer

        Returns
        -------
        outputs : TensorFlow Tensor
            The outputs of the layer's logic
        """
        if self.is_on:
            return inputs
        else:
            return tf.zeros_like(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'is_on': self.is_on})
        return config

    def turn_on(self):
        """Turn the layer `on` so inputs are returned unchanged as outputs"""
        self.is_on = True

    def turn_off(self):
        """Turn the layer `off` so inputs are destroyed and all-zero tensors are output"""
        self.is_on = False

    @classmethod
    def from_config(cls, config):
        return cls(**config)
