import pytest
import mann
import tensorflow as tf
import numpy as np
import os


def build_model():
    input1 = tf.keras.layers.Input((10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 3))
    x = mann.layers.MultiMaskedConv2D(
        8,
        activation='relu'
    )([input1, input2])
    x = mann.layers.MultiConv2D(
        8,
        activation='relu'
    )(x)
    x = mann.layers.MultiMaxPool2D()(x)
    x1 = mann.layers.SelectorLayer(0)(x)
    x2 = mann.layers.SelectorLayer(1)(x)
    x1 = mann.layers.MaskedConv2D(8)(x1)
    x2 = mann.layers.MaskedConv2D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = mann.layers.MaskedDense(10)(x1)
    x2 = mann.layers.MaskedDense(10)(x2)
    x = mann.layers.MultiMaskedDense(10)([x1, x2])
    x = mann.layers.MultiDense(10)(x)
    x1 = mann.layers.SelectorLayer(0)(x)
    x2 = mann.layers.SelectorLayer(1)(x)
    x1 = mann.layers.FilterLayer()(x1)
    x2 = mann.layers.FilterLayer(False)(x2)
    out1 = x1
    out2 = x2
    out3 = mann.layers.SumLayer()([x1, x2])
    model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2, out3]
    )
    return model


def build_simple_model():
    input1 = tf.keras.layers.Input((10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 3))
    x = mann.layers.MultiMaskedConv2D(
        8,
        activation='relu'
    )([input1, input2])
    x = mann.layers.MultiConv2D(
        8,
        activation='relu'
    )(x)
    x = mann.layers.MultiMaxPool2D()(x)
    x1 = mann.layers.SelectorLayer(0)(x)
    x2 = mann.layers.SelectorLayer(1)(x)
    x1 = mann.layers.MaskedConv2D(8)(x1)
    x2 = mann.layers.MaskedConv2D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = mann.layers.MaskedDense(10)(x1)
    x2 = mann.layers.MaskedDense(10)(x2)
    x = mann.layers.MultiMaskedDense(10)([x1, x2])
    x = mann.layers.MultiDense(10)(x)
    x1 = mann.layers.SelectorLayer(0)(x)
    x2 = mann.layers.SelectorLayer(1)(x)
    out1 = x1
    out2 = x2
    model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2]
    )
    return model


def test_tf_layer_functionality():
    model = build_model()
    preds = model.predict(
        [np.random.random((100, 10, 10, 3))] * 2
    )
    assert np.isclose(preds[1], 0).all()
    assert (preds[0] == preds[2]).all()


def test_save_load(tmp_path):
    model = build_model()
    save_path = os.path.join(tmp_path, 'test.h5')
    model.save(save_path)
    tf.keras.models.load_model(
        save_path,
        custom_objects=mann.utils.get_custom_objects()
    )


def test_pruning():
    model = build_simple_model()
    model.compile(loss='mse', optimizer='adam')
    model = mann.utils.mask_model(
        model,
        70,
        x=[np.random.random((100, 10, 10, 3))] * 2,
        y=[np.random.random((100, 10))] * 2
    )
    for layer in model.layers:
        if isinstance(layer, mann.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, mann.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5

    model = build_simple_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(
        [np.random.random((1000, 10, 10, 3))] * 2,
        [np.random.random((1000, 10))] * 2
    )
    model = mann.utils.mask_model(
        model,
        70,
        method='magnitude'
    )
    for layer in model.layers:
        if isinstance(layer, mann.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, mann.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5


def test_remove_layer_masks():
    model = build_model()
    model.compile(loss='mse', optimizer='adam')
    to_pred = [np.random.random((100, 10, 10, 3))] * 2
    og_preds = model.predict(to_pred)
    converted_model = mann.utils.remove_layer_masks(model)
    new_preds = converted_model.predict(to_pred)
    assert np.isclose(og_preds[0], new_preds[0]).all()
    assert np.isclose(og_preds[1], new_preds[1]).all()


def test_add_layer_masks():
    model = build_model()
    model.compile(loss='mse', optimizer='adam')
    to_pred = [np.random.random((100, 10, 10, 3))] * 2
    model = mann.utils.remove_layer_masks(model)
    og_preds = model.predict(to_pred)
    converted_model = mann.utils.add_layer_masks(model)
    new_preds = converted_model.predict(to_pred)
    assert np.isclose(og_preds[0], new_preds[0]).all()
    assert np.isclose(og_preds[1], new_preds[1]).all()
