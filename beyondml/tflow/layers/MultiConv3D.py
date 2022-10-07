from multiprocessing.sharedctypes import Value
import tensorflow as tf
from tensorflow.keras.layers import Layer


class MultiConv3D(Layer):
    """
    Multitask 3-dimensional convolutional layer

    This layer implements multiple stacks of convolutional weights to account for different ways individual
    neurons activate for various tasks. It is expected that to train using the RSN2 algorithm that MultiMaskedConv3D
    layers be used during training and then those layers be converted to this layer type.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        padding='same',
        strides=1,
        use_bias=True,
        activation=None,
        kernel_initializer='random_normal',
        bias_initializer='zeros',
        **kwargs
    ):
        """
        Parameters
        ----------
        filters : int
            The number of convolutional filters to apply
        kernel_size : int or tuple of ints (default 3)
            The kernel size in height and width
        padding : str (default 'same')
            Either 'same' or 'valid', the padding to use during convolution
        strides : int or tuple of ints
            Stride lengths to use during convolution
        use_bias : bool (default True)
            Whether to use a bias calculation on the outputs
        activation : None, str, or function (default None)
            Activation function to use on the outputs
        kernel_initializer : str or keras initialization function (default 'random_normal')
            The weight initialization function to use
        bias_initializer : str or keras initialization function (default 'zeros')
            The bias initialization function to use

        """
        super().__init__(**kwargs)
        self.filters = int(filters) if not isinstance(
            filters, int) else filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        if isinstance(value, int):
            self._kernel_size = (value, value, value)
        else:
            self._kernel_size = value

    def build(self, input_shape):
        """
        Build the layer in preparation to be trained or called. Should not be called directly,
        but rather is called when the layer is added to a model
        """
        input_shape = [
            tuple(shape.as_list()) for shape in input_shape
        ]
        if len(set(input_shape)) != 1:
            raise ValueError(
                f'All input shapes must be equal, got {input_shape}'
            )

        simplified_shape = input_shape[0]

        self.w = self.add_weight(
            shape=(len(input_shape), self.kernel_size[0], self.kernel_size[1],
                   self.kernel_size[2], simplified_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            name='weights'
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape=(len(input_shape), self.filters),
                initializer=self.bias_initializer,
                trainable=True,
                name='bias'
            )

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
        conv_outputs = [
            tf.nn.convolution(
                inputs[i],
                self.w[i],
                padding=self.padding.upper() if isinstance(
                    self.padding, str) else self.padding,
                strides=self.strides,
                data_format='NDHWC'
            ) for i in range(len(inputs))
        ]
        if self.use_bias:
            conv_outputs = [
                conv_outputs[i] + self.b[i] for i in range(len(conv_outputs))
            ]
        return [self.activation(output) for output in conv_outputs]

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'padding': self.padding,
                'strides': self.strides,
                'use_bias': self.use_bias,
                'activation': tf.keras.activations.serialize(self.activation),
                'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer)
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides'],
            activation=config['activation'],
            use_bias=config['use_bias'],
            kernel_initializer=config['kernel_initializer'],
            bias_initializer=config['bias_initializer']
        )
