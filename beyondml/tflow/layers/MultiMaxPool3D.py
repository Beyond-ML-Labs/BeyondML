import tensorflow as tf
from tensorflow.keras.layers import Layer


class MultiMaxPool3D(Layer):
    """
    Multitask 3D Max Pooling Layer. This layer implements the Max Pooling
    algorithm across multiple inputs for developing multitask models
    """

    def __init__(
        self,
        pool_size=(3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        **kwargs
    ):
        """
        Parameters
        ----------
        pool_size : integer or tuple of 3 integers (default (3, 3, 3))
            Window size over which to take the maximum
        strides : integer or tuple of 3 integers (default (1, 1, 1))
            Stride values to move the pooling window after each step
        padding : str (default 'same')
            One of either 'same' or 'valid', case-insensitive. The
            padding to apply to the inputs

        """
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

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
        return [
            tf.nn.max_pool3d(
                input=inputs[i],
                ksize=self.pool_size,
                strides=self.strides,
                padding=self.padding.upper(),
                data_format='NDHWC'
            ) for i in range(len(inputs))
        ]

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'pool_size': self.pool_size,
                'strides': self.strides,
                'padding': self.padding
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            pool_size=config['pool_size'],
            strides=config['strides'],
            padding=config['padding']
        )
