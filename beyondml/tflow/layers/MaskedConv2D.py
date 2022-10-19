import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class MaskedConv2D(Layer):
    """
    Masked 2-dimensional convolutional layer. For full documentation of the convolutional architecture, see the
    TensorFlow Keras Convolutional2D layer documentation.

    This layer implements masking consistent with the BeyondML API to support developing sparse models.

    """

    def __init__(
            self,
            filters,
            kernel_size=3,
            padding='same',
            strides=1,
            activation=None,
            use_bias=True,
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
        strides : int or tuple of ints (default 1)
            Stride lengths to use during convolution
        activation : None, str, or function (default None)
            Activation function to use on the outputs
        use_bias : bool (default True)
            Whether to use a bias calculation on the outputs
        kernel_initializer : str or keras initialization function (default 'random_normal')
            The weight initialization function to use
        bias_initializer : str or keras initialization function (default 'zeros')
            The bias initialization function to use
        mask_initializer : str or keras initialization function (default 'ones')
            The mask initialization function to use

    """
        super(MaskedConv2D, self).__init__(**kwargs)
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
        self.w = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1],
                   input_shape[-1], self.filters),
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
                shape=(self.filters,),
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
        conv_output = tf.nn.convolution(
            inputs,
            self.w * self.w_mask,
            padding=self.padding.upper() if isinstance(
                self.padding, str) else self.padding,
            strides=self.strides,
            data_format='NHWC'
        )
        if self.use_bias:
            conv_output = conv_output + (self.b * self.b_mask)
        return self.activation(conv_output)

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
        """
        Set the masks for the layer

        Parameters
        ----------
        new_masks : list of arrays or array-likes
            The new masks to set for the layer
        """
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
