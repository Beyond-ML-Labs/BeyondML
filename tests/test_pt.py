import pytest
import beyondml.pt as pt
import torch
import numpy as np


def test_conv2d():
    layer = pt.layers.Conv2D(
        np.random.random((8, 3, 3, 3)),
        np.random.random(8)
    )
    forward = layer.forward(torch.Tensor(10, 3, 30, 30))
    assert forward.shape == (10, 8, 30, 30)


def test_conv3d():
    layer = pt.layers.Conv3D(
        np.random.random((8, 3, 3, 3, 3)),
        np.random.random(8)
    )
    forward = layer.forward(torch.Tensor(10, 3, 30, 30, 30))
    assert forward.shape == (10, 8, 30, 30, 30)


def test_dense():
    layer = pt.layers.Dense(
        np.random.random((10, 100)),
        np.random.random(100)
    )
    forward = layer.forward(torch.Tensor(1000, 10))
    assert forward.shape == (1000, 100)


def test_sparse_conv2d():
    layer = pt.layers.SparseConv2D(
        np.random.random((8, 3, 3, 3)),
        np.random.random(8)
    )
    forward = layer.forward(torch.Tensor(10, 3, 30, 30))
    assert forward.shape == (10, 8, 30, 30)


def test_sparse_conv3d():
    layer = pt.layers.SparseConv3D(
        np.random.random((8, 3, 3, 3, 3)),
        np.random.random(8)
    )
    forward = layer.forward(torch.Tensor(10, 3, 30, 30, 30))
    assert forward.shape == (10, 8, 30, 30, 30)


def test_sparse_dense():
    layer = pt.layers.SparseDense(
        np.random.random((10, 100)),
        np.random.random(100)
    )
    forward = layer.forward(torch.Tensor(1000, 10))
    assert forward.shape == (1000, 100)


def test_multi_conv2d():
    layer = pt.layers.MultiConv2D(
        np.random.random((2, 8, 3, 3, 3)),
        np.random.random((2, 8))
    )
    forward = layer.forward([torch.Tensor(10, 3, 30, 30)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30)


def test_multi_conv3d():
    layer = pt.layers.MultiConv3D(
        np.random.random((2, 8, 3, 3, 3, 3)),
        np.random.random((2, 8))
    )
    forward = layer.forward([torch.Tensor(10, 3, 30, 30, 30)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30, 30)


def test_multi_dense():
    layer = pt.layers.MultiDense(
        np.random.random((2, 10, 100)),
        np.random.random((2, 100))
    )
    forward = layer.forward([torch.Tensor(10, 10)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 100)
    assert forward[1].shape == (10, 100)


def test_sparse_multi_conv2d():
    layer = pt.layers.SparseMultiConv2D(
        np.random.random((2, 8, 3, 3, 3)),
        np.random.random((2, 8))
    )
    forward = layer.forward([torch.Tensor(10, 3, 30, 30)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30)


def test_sparse_multi_conv3d():
    layer = pt.layers.SparseMultiConv3D(
        np.random.random((2, 8, 3, 3, 3, 3)),
        np.random.random((2, 8))
    )
    forward = layer.forward([torch.Tensor(10, 3, 30, 30, 30)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30, 30)


def test_sparse_multi_dense():
    layer = pt.layers.SparseMultiDense(
        np.random.random((2, 10, 100)),
        np.random.random((2, 100))
    )
    forward = layer.forward([torch.Tensor(10, 10)] * 2)
    assert len(forward) == 2
    assert forward[0].shape == (10, 100)
    assert forward[1].shape == (10, 100)


def test_filter_layer():
    layer = pt.layers.FilterLayer()
    rand = torch.Tensor(10, 10)
    forward = layer.forward(rand)
    assert np.allclose(forward, rand, 1e-1, 1e-3)

    layer.turn_off()
    forward = layer.forward(rand)
    assert np.allclose(forward, torch.zeros_like(rand))

    layer.turn_on()
    forward = layer.forward(rand)
    assert np.allclose(forward, rand, 1e-1, 1e-3)


def test_masked_conv2d():
    layer = pt.layers.MaskedConv2D(
        3,
        8
    )
    forward = layer.forward(
        torch.Tensor(10, 3, 30, 30)
    )
    assert forward.shape == (10, 8, 30, 30)

    layer.prune(60)
    assert 1 - layer.w_mask.sum().numpy() / \
        layer.w_mask.flatten().shape[0] <= 0.6


def test_masked_conv3d():
    layer = pt.layers.MaskedConv3D(
        3,
        8
    )
    forward = layer.forward(
        torch.Tensor(10, 3, 30, 30, 30)
    )
    assert forward.shape == (10, 8, 30, 30, 30)

    layer.prune(60)
    assert 1 - layer.w_mask.numpy().sum() / \
        layer.w_mask.flatten().shape[0] <= 0.605


def test_masked_dense():
    layer = pt.layers.MaskedDense(
        10,
        100
    )
    forward = layer.forward(
        torch.Tensor(1000, 10)
    )
    assert forward.shape == (1000, 100)

    layer.prune(60)
    assert 1 - layer.w_mask.sum().numpy() / \
        layer.w_mask.flatten().shape[0] <= 0.6


def test_multi_masked_conv2d():
    layer = pt.layers.MultiMaskedConv2D(
        3,
        8,
        2
    )
    forward = layer.forward(
        [torch.Tensor(10, 3, 30, 30)] * 2
    )
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30)

    layer.prune(60)
    assert 1 - layer.w_mask[0].sum().numpy() / \
        layer.w_mask[0].flatten().shape[0] <= 0.6
    assert 1 - layer.w_mask[1].sum().numpy() / \
        layer.w_mask[1].flatten().shape[0] <= 0.6
    assert layer.w_mask.sum(axis=0).max() < 2


def test_multi_masked_conv3d():
    layer = pt.layers.MultiMaskedConv3D(
        3,
        8,
        2
    )
    forward = layer.forward(
        [torch.Tensor(10, 3, 30, 30, 30)] * 2
    )
    assert len(forward) == 2
    assert forward[0].shape == (10, 8, 30, 30, 30)
    assert forward[1].shape == (10, 8, 30, 30, 30)

    layer.prune(60)
    assert 1 - layer.w_mask[0].sum().numpy() / \
        layer.w_mask[0].flatten().shape[0] <= 0.65
    assert 1 - layer.w_mask[1].sum().numpy() / \
        layer.w_mask[1].flatten().shape[0] <= 0.65
    assert layer.w_mask.sum(axis=0).max() < 2


def test_multi_masked_dense():
    layer = pt.layers.MultiMaskedDense(
        10,
        100,
        2
    )
    forward = layer.forward(
        [torch.Tensor(1000, 10)] * 2
    )
    assert len(forward) == 2
    assert forward[0].shape == (1000, 100)
    assert forward[1].shape == (1000, 100)

    layer.prune(60)
    assert 1 - layer.w_mask[0].sum().numpy() / \
        layer.w_mask[0].flatten().shape[0] <= 0.6
    assert 1 - layer.w_mask[1].sum().numpy() / \
        layer.w_mask[1].flatten().shape[0] <= 0.6
    assert layer.w_mask.sum(axis=0).max() < 2


def test_prune():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = pt.layers.MaskedDense(10, 10)
            self.layer2 = pt.layers.MaskedDense(10, 10)

        @property
        def layers(self):
            return [self.layer1, self.layer2]

        def forward(self, inputs):
            return self.layer2.forward(
                self.layer1.forward(
                    inputs
                )
            )

    model = SimpleModel()
    model = pt.utils.prune_model(model, 90)
    assert np.isclose(
        model.layer1.w_mask.sum() / model.layer1.w_mask.flatten().shape[0],
        0.1
    )
    assert np.isclose(
        model.layer2.w_mask.sum() / model.layer2.w_mask.flatten().shape[0],
        0.1
    )


def test_transformer():
    encoder = pt.layers.MaskedTransformerEncoderLayer(
        512,
        8
    )
    encoder.prune(80)
    decoder = pt.layers.MaskedTransformerDecoderLayer(
        2048,
        8
    )
    decoder.prune(80)
