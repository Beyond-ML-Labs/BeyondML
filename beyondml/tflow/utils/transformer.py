import tensorflow as tf
from beyondml.tflow.layers import MultiDense, SelectorLayer


def build_transformer_block(
        input_shape,
        embed_dim,
        num_heads,
        neurons,
        dropout_rate=0.1,
):
    """
    Build a Transformer Block

    Parameters
    ----------
    input_shape : int or tuple of int
        The input shape for the model to use
    embed_dim : int
        The dimension of the embedding
    num_heads : int
        The number of attention heads to use
    neurons : int
        The number of hidden neurons to use in the hidden layer
    dropout_rate : float (default 0.1)
        Rate at which dropout is applied
    value_dim : int or None (default None)
        The dimension to use for the `value` matrix, if provided

    Returns
    -------
    transformer_block : TensorFlow keras Functional model
        The transformer block, which can then be used alone or as
        a layer in another model
    """
    input_layer = tf.keras.layers.Input(input_shape)
    query = MultiDense(embed_dim)([input_layer] * num_heads)
    key = MultiDense(embed_dim)([input_layer] * num_heads)
    value = MultiDense(embed_dim)([input_layer] * num_heads)

    query_selectors = [
        SelectorLayer(i)(query) for i in range(num_heads)
    ]
    key_selectors = [
        SelectorLayer(i)(key) for i in range(num_heads)
    ]
    value_selectors = [
        SelectorLayer(i)(value) for i in range(num_heads)
    ]
    attention_layers = [
        tf.keras.layers.Attention()([query_selectors[i], key_selectors[i], value_selectors[i]]) for i in range(num_heads)
    ]
    concat = tf.keras.layers.Concatenate()(attention_layers)
    merge = tf.keras.layers.Reshape((input_shape[0], -1))(concat)

    x = tf.keras.layers.Dropout(dropout_rate)(merge)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(neurons, activation='relu')(out1)
    x = tf.keras.layers.Dense(embed_dim * num_heads)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([out1, x])
    output_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    return tf.keras.models.Model(input_layer, output_layer)


def build_token_position_embedding_block(
    sequence_length,
    vocab_size,
    embed_dim
):
    """
    Builds a token and position embedding block

    Parameters
    ----------
    sequence_length : int
        The length of each sequence
    vocab_size : int
        The size of the vocabulary used
    embed_dim : int
        The desired embedding dimension

    Returns
    -------
    embedding_block : TensorFlow keras Functional model
        The embedding block, which can be used alone or
        as a layer in another model
    """
    tok_input = tf.keras.layers.Input(sequence_length)
    pos_input = tf.keras.layers.Input(sequence_length)

    tok_embed = tf.keras.layers.Embedding(
        vocab_size, output_dim=embed_dim)(tok_input)
    pos_embed = tf.keras.layers.Embedding(
        sequence_length, output_dim=embed_dim)(pos_input)
    output_layer = tf.keras.layers.Add()([tok_embed, pos_embed])

    return tf.keras.models.Model([tok_input, pos_input], output_layer)
