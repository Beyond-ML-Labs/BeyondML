import tensorflow as tf
from mann.layers import MaskedDense, MultiMaskedDense

def build_transformer_block(
    input_shape,
    embed_dim,
    num_heads,
    neurons,
    dropout_rate = 0.1
):
    input_layer = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads = num_heads,
        key_dim = embed_dim
    )(input_layer, input_layer)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(x)
    x = tf.keras.layers.Dense(neurons, activation = 'relu')(out1)
    x = tf.keras.layers.Dense(embed_dim)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Add()([out1, x])
    output_layer = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(x)
    
    return tf.keras.models.Model(input_layer, output_layer)

def build_token_position_embedding_block(
    sequence_length,
    vocab_size,
    embed_dim
):
    tok_input = tf.keras.layers.Input(sequence_length)
    pos_input = tf.keras.layers.Input(sequence_length)

    tok_embed = tf.keras.layers.Embedding(vocab_size, output_dim = embed_dim)(tok_input)
    pos_embed = tf.keras.layers.Embedding(sequence_length, output_dim = embed_dim)(pos_input)
    output_layer = tf.keras.layers.Add()([tok_embed, pos_embed])
    
    return tf.keras.models.Model([tok_input, pos_input], output_layer)

