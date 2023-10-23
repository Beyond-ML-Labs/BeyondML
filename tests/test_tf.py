import pytest
import beyondml
import tensorflow as tf
import numpy as np
import os


def build_model():
    input1 = tf.keras.layers.Input((10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 3))
    x = beyondml.tflow.layers.MultiMaskedConv2D(
        8,
        activation='relu'
    )([input1, input2])
    x = beyondml.tflow.layers.MultiConv2D(
        8,
        activation='relu'
    )(x)
    x = beyondml.tflow.layers.MultiMaxPool2D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.MaskedConv2D(8)(x1)
    x2 = beyondml.tflow.layers.MaskedConv2D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.MaskedDense(10)(x1)
    x2 = beyondml.tflow.layers.MaskedDense(10)(x2)
    x = beyondml.tflow.layers.MultiMaskedDense(10)([x1, x2])
    x = beyondml.tflow.layers.MultiDense(10)(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.FilterLayer()(x1)
    x2 = beyondml.tflow.layers.FilterLayer(False)(x2)
    out1 = x1
    out2 = x2
    out3 = beyondml.tflow.layers.SumLayer()([x1, x2])
    model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2, out3]
    )
    return model


def build_simple_model():
    input1 = tf.keras.layers.Input((10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 3))
    x = beyondml.tflow.layers.MultiMaskedConv2D(
        8,
        activation='relu'
    )([input1, input2])
    x = beyondml.tflow.layers.MultiConv2D(
        8,
        activation='relu'
    )(x)
    x = beyondml.tflow.layers.MultiMaxPool2D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.MaskedConv2D(8)(x1)
    x2 = beyondml.tflow.layers.MaskedConv2D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.MaskedDense(10)(x1)
    x2 = beyondml.tflow.layers.MaskedDense(10)(x2)
    x = beyondml.tflow.layers.MultiMaskedDense(10)([x1, x2])
    x = beyondml.tflow.layers.MultiDense(10)(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    out1 = x1
    out2 = x2
    model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2]
    )
    return model


def build_3d_model():
    input1 = tf.keras.layers.Input((10, 10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 10, 3))
    x = beyondml.tflow.layers.MultiMaskedConv3D(
        8,
        activation='relu'
    )([input1, input2])
    x = beyondml.tflow.layers.MultiConv3D(
        8,
        activation='relu'
    )(x)
    x = beyondml.tflow.layers.MultiMaxPool3D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.MaskedConv3D(8)(x1)
    x2 = beyondml.tflow.layers.MaskedConv3D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.MaskedDense(10)(x1)
    x2 = beyondml.tflow.layers.MaskedDense(10)(x2)
    x = beyondml.tflow.layers.MultiMaskedDense(10)([x1, x2])
    x = beyondml.tflow.layers.MultiDense(10)(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.FilterLayer()(x1)
    x2 = beyondml.tflow.layers.FilterLayer(False)(x2)
    out1 = x1
    out2 = x2
    out3 = beyondml.tflow.layers.SumLayer()([x1, x2])
    model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2, out3]
    )
    return model


def build_simple_3d_model():
    input1 = tf.keras.layers.Input((10, 10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 10, 3))
    x = beyondml.tflow.layers.MultiMaskedConv3D(
        8,
        activation='relu'
    )([input1, input2])
    x = beyondml.tflow.layers.MultiConv3D(
        8,
        activation='relu'
    )(x)
    x = beyondml.tflow.layers.MultiMaxPool3D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.MaskedConv3D(8)(x1)
    x2 = beyondml.tflow.layers.MaskedConv3D(8)(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.MaskedDense(10)(x1)
    x2 = beyondml.tflow.layers.MaskedDense(10)(x2)
    x = beyondml.tflow.layers.MultiMaskedDense(10)([x1, x2])
    x = beyondml.tflow.layers.MultiDense(10)(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
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
    save_path = os.path.join(tmp_path, 'test.keras')
    model.save(save_path)
    tf.keras.models.load_model(
        save_path,
        custom_objects=beyondml.tflow.utils.get_custom_objects()
    )


def test_pruning():
    model = build_simple_model()
    model.compile(loss='mse', optimizer='adam')
    model = beyondml.tflow.utils.mask_model(
        model,
        70,
        x=[np.random.random((100, 10, 10, 3))] * 2,
        y=[np.random.random((100, 10))] * 2
    )
    for layer in model.layers:
        if isinstance(layer, beyondml.tflow.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, beyondml.tflow.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5

    model = build_simple_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(
        [np.random.random((1000, 10, 10, 3))] * 2,
        [np.random.random((1000, 10))] * 2
    )
    model = beyondml.tflow.utils.mask_model(
        model,
        70,
        method='magnitude'
    )
    for layer in model.layers:
        if isinstance(layer, beyondml.tflow.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, beyondml.tflow.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5

    model = build_simple_3d_model()
    model.compile(loss='mse', optimizer='adam')
    model = beyondml.tflow.utils.mask_model(
        model,
        70,
        x=[np.random.random((100, 10, 10, 10, 3))] * 2,
        y=[np.random.random((100, 10))] * 2
    )
    for layer in model.layers:
        if isinstance(layer, beyondml.tflow.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, beyondml.tflow.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5

    model = build_simple_3d_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit(
        [np.random.random((1000, 10, 10, 10, 3))] * 2,
        [np.random.random((1000, 10))] * 2
    )
    model = beyondml.tflow.utils.mask_model(
        model,
        70,
        method='magnitude'
    )
    for layer in model.layers:
        if isinstance(layer, beyondml.tflow.utils.utils.MULTI_MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum(axis=0).max() == 1
        elif isinstance(layer, beyondml.tflow.utils.utils.MASKING_LAYERS):
            for weight in layer.get_weights()[2:]:
                assert weight.sum() / weight.flatten().shape[0] <= 0.5


# def test_remove_layer_masks():
    # model = build_model()
    # model.compile(loss='mse', optimizer='adam')
    # to_pred = [np.random.random((100, 10, 10, 3))] * 2
    # og_preds = model.predict(to_pred)
    # converted_model = beyondml.tflow.utils.remove_layer_masks(model)
    # new_preds = converted_model.predict(to_pred)
    # assert np.isclose(og_preds[0], new_preds[0]).all()
    # assert np.isclose(og_preds[1], new_preds[1]).all()

    # model = build_simple_3d_model()
    # model.compile(loss='mse', optimizer='adam')
    # to_pred = [np.random.random((100, 10, 10, 10, 3))] * 2
    # og_preds = model.predict(to_pred)
    # converted_model = beyondml.tflow.utils.remove_layer_masks(model)
    # new_preds = converted_model.predict(to_pred)
    # assert np.isclose(og_preds[0], new_preds[0]).all()
    # assert np.isclose(og_preds[1], new_preds[1]).all()


# def test_add_layer_masks():
    # model = build_model()
    # model.compile(loss='mse', optimizer='adam')
    # to_pred = [np.random.random((100, 10, 10, 3))] * 2
    # model = beyondml.tflow.utils.remove_layer_masks(model)
    # og_preds = model.predict(to_pred)
    # converted_model = beyondml.tflow.utils.add_layer_masks(model)
    # new_preds = converted_model.predict(to_pred)
    # assert np.isclose(og_preds[0], new_preds[0]).all()
    # assert np.isclose(og_preds[1], new_preds[1]).all()

    # model = build_simple_3d_model()
    # model.compile(loss='mse', optimizer='adam')
    # to_pred = [np.random.random((100, 10, 10, 10, 3))] * 2
    # model = beyondml.tflow.utils.remove_layer_masks(model)
    # og_preds = model.predict(to_pred)
    # converted_model = beyondml.tflow.utils.add_layer_masks(model)
    # new_preds = converted_model.predict(to_pred)
    # assert np.isclose(og_preds[0], new_preds[0]).all()
    # assert np.isclose(og_preds[1], new_preds[1]).all()


def test_quantize():
    model = build_simple_model()
    new_model = beyondml.tflow.utils.quantize_model(model)
    to_pred = [np.random.random((100, 10, 10, 3))] * 2
    og_preds = model.predict(to_pred)
    new_preds = new_model.predict(to_pred)
    assert all(
        [
            np.allclose(new_preds[i], og_preds[i], 1e-3, 1e-5)
            for i in range(2)
        ]
    )


def test_sparse():
    model = build_model()
    model.compile(loss='mse', optimizer='adam')

    input1 = tf.keras.layers.Input((10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 3))
    x = beyondml.tflow.layers.SparseMultiConv2D.from_layer(
        model.layers[2])([input1, input2])
    x = beyondml.tflow.layers.SparseMultiConv2D.from_layer(model.layers[3])(x)
    x = beyondml.tflow.layers.MultiMaxPool2D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.SparseConv2D.from_layer(model.layers[7])(x1)
    x2 = beyondml.tflow.layers.SparseConv2D.from_layer(model.layers[8])(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.SparseDense.from_layer(model.layers[11])(x1)
    x2 = beyondml.tflow.layers.SparseDense.from_layer(model.layers[12])(x2)
    x = beyondml.tflow.layers.SparseMultiDense.from_layer(
        model.layers[13])([x1, x2])
    x = beyondml.tflow.layers.SparseMultiDense.from_layer(model.layers[14])(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.FilterLayer()(x1)
    x2 = beyondml.tflow.layers.FilterLayer(False)(x2)
    out1 = x1
    out2 = x2
    out3 = beyondml.tflow.layers.SumLayer()([x1, x2])
    sparse_model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2, out3]
    )

    model = build_3d_model()
    model.compile(loss='mse', optimizer='adam')

    input1 = tf.keras.layers.Input((10, 10, 10, 3))
    input2 = tf.keras.layers.Input((10, 10, 10, 3))
    x = beyondml.tflow.layers.SparseMultiConv3D.from_layer(
        model.layers[2])([input1, input2])
    x = beyondml.tflow.layers.SparseMultiConv3D.from_layer(model.layers[3])(x)
    x = beyondml.tflow.layers.MultiMaxPool3D()(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.SparseConv3D.from_layer(model.layers[7])(x1)
    x2 = beyondml.tflow.layers.SparseConv3D.from_layer(model.layers[8])(x2)
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x1 = beyondml.tflow.layers.SparseDense.from_layer(model.layers[11])(x1)
    x2 = beyondml.tflow.layers.SparseDense.from_layer(model.layers[12])(x2)
    x = beyondml.tflow.layers.SparseMultiDense.from_layer(
        model.layers[13])([x1, x2])
    x = beyondml.tflow.layers.SparseMultiDense.from_layer(model.layers[14])(x)
    x1 = beyondml.tflow.layers.SelectorLayer(0)(x)
    x2 = beyondml.tflow.layers.SelectorLayer(1)(x)
    x1 = beyondml.tflow.layers.FilterLayer()(x1)
    x2 = beyondml.tflow.layers.FilterLayer(False)(x2)
    out1 = x1
    out2 = x2
    out3 = beyondml.tflow.layers.SumLayer()([x1, x2])
    sparse_model = tf.keras.models.Model(
        [input1, input2],
        [out1, out2, out3]
    )
