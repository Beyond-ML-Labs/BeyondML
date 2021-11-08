import tensorflow as tf
from tensorflow.keras.layers import Layer

class SumLayer(Layer):
    """
    Layer which adds all inputs together. All inputs must have compatible shapes

    Example:

    >>> # Create a model with just a SumLayer and two inputs
    >>> input_1 = tf.keras.layers.Input(10)
    >>> input_2 = tf.keras.layers.Input(10)
    >>> sum_layer = mann.layers.SumLayer()([input_1, input_2])
    >>> model = tf.keras.models.Model([input_1, input_2], sum_layer)
    >>> model.compile()
    >>> # Call the model
    >>> data = np.arange(10).reshape((1, 10))
    >>> model.predict([data, data])
    array([[ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.]], dtype=float32)
    
    """
    def __init__(self, **kwargs):
        super(SumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.add_n(inputs)

    def get_config(self):
        return super().get_config().copy()
