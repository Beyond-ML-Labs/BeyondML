import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MultiDense(Layer):
    """
    Multitask fully connected layer

    This layer implements multiple stacks of fully connected weights to account for different
    ways neurons can activate for various tasks. It is expected that to train using the RSN2 algorithm
    that MultiMaskedDense layers be used during training and then those layers be converted to this layer type.

    """
    def __init__(
        self,
        units,
        use_bias = True,
        activation = None,
        kernel_initializer = 'random_normal',
        bias_initializer = 'zeros',
        **kwargs
    ):
        """
        Parameters
        ----------
        units : int
            The number of artificial neurons to use
        use_bias : bool (default True)
            Whether to use a bias calculation in the outputs
        activation : None, str, or function (default None)
            The activation function to use on the outputs
        kernel_initializer : str or keras initialization function (default 'random_normal')
            The weight initialization function to use
        bias_initializer : str or keras initialization function (default 'zeros')
            The bias initialization function to use

        """
        super(MultiDense, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = [
            tuple(shape.as_list()) for shape in input_shape
        ]
        if len(set(input_shape)) != 1:
            raise ValueError(f'All input shapes must be equal, got {input_shape}')

        simplified_shape = input_shape[0]

        self.w = self.add_weight(
            shape = (len(input_shape), simplified_shape[-1], self.units),
            initializer = self.kernel_initializer,
            trainable = True,
            name = 'weights'
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape = (len(input_shape), self.units),
                initializer = self.bias_initializer,
                trainable = True,
                name = 'bias'
            )

    def call(self, inputs):
        output_tensor = [
            tf.matmul(inputs[i], self.w[i]) for i in range(len(inputs))
        ]
        if self.use_bias:
            output_tensor = [
                output_tensor[i] + (self.b[i]) for i in range(len(output_tensor))
            ]
        return [self.activation(tensor) for tensor in output_tensor]

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'units' : self.units,
                'use_bias' : self.use_bias,
                'activation' : tf.keras.activations.serialize(self.activation),
                'kernel_initializer' : tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer' : tf.keras.initializers.serialize(self.bias_initializer)
            }
        )
        return config
