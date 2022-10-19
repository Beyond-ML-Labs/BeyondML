import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class MultiMaskedConv2D(Layer):
    """
    Masked multitask 2-dimensional convolutional layer. This layer implements
    multiple stacks of the convolutional architecture and implements masking consistent
    with the BeyondML API to support developing sparse multitask models.
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
        mask_initializer='ones',
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
            Stride lenghts to use during convolution
        use_bias : bool (default True)
            Whether to use a bias calculation on the outputs
        activation : None, str, or function (default None)
            Activation function to use on the outputs
        kernel_initializer : str or keras initialization function (default 'random_normal')
            The initialization function to use for the weights
        bias_initializer : str or keras initialization function (default 'zeros')
            The initialization function to use for the bias
        mask_initializer : str or keras initialization function (default 'ones')
            The mask initialization function to use

        """
        super(MultiMaskedConv2D, self).__init__(**kwargs)
        self.filters = int(filters) if not isinstance(
            filters, int) else filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.mask_initializer = tf.keras.initializers.get(mask_initializer)

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, value):
        if isinstance(value, int):
            self._kernel_size = (value, value)
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
                f'All input shapes must be equal, got {input_shape}')

        simplified_shape = input_shape[0]

        self.w = self.add_weight(
            shape=(len(input_shape),
                   self.kernel_size[0], self.kernel_size[1], simplified_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
            name='weights'
        )
        self.w_mask = self.add_weight(
            shape=self.w.shape,
            initializer=self.mask_initializer,
            trainable=False,
            name='weights_mask'
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape=(len(input_shape), self.filters),
                initializer=self.bias_initializer,
                trainable=True,
                name='bias'
            )
            self.b_mask = self.add_weight(
                shape=self.b.shape,
                initializer=self.mask_initializer,
                trainable=False,
                name='bias_mask'
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
                self.w[i] * self.w_mask[i],
                padding=self.padding.upper(),
                strides=self.strides,
                data_format='NHWC'
            ) for i in range(len(inputs))
        ]
        if self.use_bias:
            conv_outputs = [
                conv_outputs[i] + (self.b[i] * self.b_mask[i]) for i in range(len(conv_outputs))
            ]
        return [self.activation(output) for output in conv_outputs]

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': list(self.kernel_size),
                'padding': self.padding,
                'strides': self.strides,
                'activation': tf.keras.activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
                'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
                'mask_initializer': tf.keras.initializers.serialize(self.mask_initializer)
            }
        )
        return config

    def set_masks(self, new_masks):
        if not self.use_bias:
            self.set_weights(
                [self.w.numpy() * new_masks[0].astype(np.float32),
                 new_masks[0].astype(np.float32)]
            )
        else:
            self.set_weights(
                [self.w.numpy() * new_masks[0].astype(np.float32), self.b.numpy() * new_masks[1].astype(
                    np.float32), new_masks[0].astype(np.float32), new_masks[1].astype(np.float32)]
            )

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
            bias_initializer=config['bias_initializer'],
            mask_initializer=config['mask_initializer']
        )
