from tensorflow.keras.layers import Layer
import tensorflow as tf


class SparseConv2D(Layer):
    """
    Sparse implementation of the Convolutional layer. If used in a model,
    must be saved and loaded via pickle
    """

    def __init__(
        self,
        filters,
        bias,
        padding='same',
        strides=1,
        activation=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        filters : tf.Tensor
            The convolutional filters
        bias : tf.Tensor
            the bias tensor
        padding : str, int, or tuple of int (default 'same')
            The padding to use
        strides : int or tuple of int (default 1)
            The strides to use
        activation : None, str, or keras activation function (default None)
            The activation function to use
        """
        super().__init__(**kwargs)
        self.w = tf.sparse.from_dense(filters)
        self.b = tf.sparse.from_dense(bias)
        self.padding = padding
        self.strides = strides
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        """
        Build the layer in preparation to be trained or called. Should not be called directly,
        but rather is called when the layer is added to a model
        """
        pass

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
            tf.sparse.to_dense(self.w),
            padding=self.padding.upper() if isinstance(
                self.padding, str) else self.padding,
            strides=self.strides,
            data_format='NHWC'
        )
        conv_output = conv_output + tf.sparse.to_dense(self.b)
        return self.activation(conv_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'padding': self.padding,
                'strides': self.strides,
                'activation': tf.keras.activations.serialize(self.activation)
            }
        )
        return config

    @classmethod
    def from_layer(cls, layer):
        """
        Create a layer from an instance of another layer
        """
        weights = layer.get_weights()
        w = weights[0]
        b = weights[1]
        padding = layer.padding
        strides = layer.strides
        activation = layer.activation
        return cls(
            w,
            b,
            padding,
            strides,
            activation
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)
