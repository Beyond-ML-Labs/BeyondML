from tensorflow.keras.layers import Layer


class MultitaskNormalization(Layer):
    """
    Multitask layer which normalizes all inputs to sum to 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
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
        s = 0
        for i in inputs:
            s += i
        return [i / s for i in inputs]

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
