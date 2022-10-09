from tensorflow.keras.layers import Layer


class SelectorLayer(Layer):
    """
    Layer which selects individual inputs

    Example:

    >>> # Create a model with two inputs and one SelectorLayer
    >>> input_1 = tf.keras.layers.Input(10)
    >>> input_2 = tf.keras.layers.Input(10)
    >>> selector = mann.layers.SelectorLayer(1)([input_1, input_2]) # 1 here indicates to select the second input and return it
    >>> model = tf.keras.models.Model([input_1, input_2], selector)
    >>> model.compile()
    >>> # Call the model
    >>> data1 = np.arange(10).reshape((1, 10))
    >>> data2 = 2*np.arange(10).reshape((1, 10))
    >>> model.predict([data1, data2])
    array([[ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.]], dtype=float32)

    """

    def __init__(
        self,
        sel_index,
        **kwargs
    ):
        """
        Parameters
        ----------
        sel_index : int
            The index of the inputs to be selected
        """
        super(SelectorLayer, self).__init__(**kwargs)
        self.sel_index = sel_index

    @property
    def sel_index(self):
        return self._sel_index

    @sel_index.setter
    def sel_index(self, value):
        if not isinstance(value, int):
            raise TypeError(
                f'sel_index must be int, got {value}, type {type(value)}')
        self._sel_index = value

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
        return inputs[self.sel_index]

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'sel_index': self.sel_index
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            sel_index=config['sel_index']
        )
