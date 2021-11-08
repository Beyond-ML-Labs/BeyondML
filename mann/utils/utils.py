from mann.layers import MaskedDense, MaskedConv2D, FilterLayer, SumLayer, SelectorLayer, MultiMaskedDense, MultiMaskedConv2D, MultiDense, MultiConv2D, MultiMaxPool2D
import tensorflow as tf
import numpy as np

MASKING_LAYERS = (MaskedDense, MaskedConv2D, MultiMaskedDense, MultiMaskedConv2D)
MULTI_MASKING_LAYERS = (MultiMaskedDense, MultiMaskedConv2D)
NON_MASKING_LAYERS = (MultiDense, MultiConv2D)
CUSTOM_LAYERS = MASKING_LAYERS + NON_MASKING_LAYERS + (FilterLayer, SumLayer, SelectorLayer, MultiMaxPool2D)

def _get_masking_gradients(
        model,
        x,
        y
):
    """
    Obtain masking layer gradients with respect to the tasks presented

    Parameters
    ----------
    model : tf.keras Model
        The model to get the gradients of
    x : np.array or array-like
        The input data
    y : np.array or array-like
        The true output

    Returns
    -------
    masking_gradients : list
        A list of gradients for the masking weights for the model
    """

    # Check outputs
    if isinstance(y, list):
        if not all([len(val.shape) > 1 for val in y]):
            raise ValueError('Error in output shapes. If any tasks have a single output, please reshape the value using the `.reshape(-1, 1)` method')
    elif not len(y.shape) > 1:
        raise ValueError('Error in output shapes. If your task has a single output, please reshape the value using the `.reshape(-1, 1)` method')
            
    # Grab the weights for the masking layers
    masking_weights = [
        layer.trainable_weights for layer in model.layers if isinstance(layer, MASKING_LAYERS)
    ]

    # Setup and obtain the losses
    losses = model.loss
    if not isinstance(losses, list):
        if callable(losses):
            losses = [losses] * len(x)
        losses = [tf.keras.losses.get(losses)] * len(x)
    else:
        losses = [tf.keras.losses.get(loss) if not callable(loss) else loss for loss in losses]

    # Grab the gradients for the specified weights
    with tf.GradientTape() as tape:
        raw_preds = model(x)
        losses = [losses[i](y[i], raw_preds[i]) for i in range(len(losses))]
        gradients = tape.gradient(losses, masking_weights)
    return gradients

def get_custom_objects():
    """Return a dictionary of custom objects (layers) to use when loading models trained using this package"""
    return dict(
        zip(
            ['MaskedDense', 'MaskedConv2D', 'MultiMaskedDense', 'MultiMaskedConv2D', 'MultiDense', 'MultiConv2D', 'FilterLayer', 'SumLayer', 'SelectorLayer', 'MultiMaxPool2D'],
            CUSTOM_LAYERS
        )
    )

def mask_model(
        model,
        percentile,
        method = 'gradients',
        exclusive = True,
        x = None,
        y = None
):
    """
    Mask the multitask model for training respective using the gradients for the tasks at hand

    Parameters
    ----------
    model : keras model with MANN masking layers
        The model to be masked
    percentile : int
        Percentile to use in masking. Any weights less than the `percentile` value will be made zero
    method : str (default 'gradients')
        One of either 'gradients' or 'magnitude' - the method for how to identify weights to mask
        If method is 'gradients', utilizes the gradients with respect to the passed x and y variables
        to identify the subnetwork to activate for each task
        If method is 'magnitude', uses the magnitude of the weights to identify the subnetwork to activate for each task
    exclusive : bool (default True)
        Whether to restrict previously-used weight indices for each task. If `True`, this identifies disjoint subsets of
        weights within the layer which perform the tasks requested.
    x : list of np.ndarray or array-like
        The training data input values, ignored if "method" is 'magnitude'
    y : list of np.ndarray or array-like
        The training data output values, ignored if "method" is 'magnitude'
    """

    # Check method
    method = method.lower()
    if method not in ['gradients', 'magnitude']:
        raise ValueError(f"method must be one of 'gradients', 'magnitude', got {method}")

    # Get the gradients
    if method == 'gradients':
        grads = _get_masking_gradients(
            model,
            x,
            y
        )
        
        # Work to identify the right weights if exclusive
        if exclusive:
            gradient_idx = 0
            for layer in model.layers:
                if isinstance(layer, MASKING_LAYERS):
                    if not isinstance(layer, MULTI_MASKING_LAYERS):
                        layer_grads = [np.abs(grad) for grad in grads[gradient_idx]]
                        new_masks = [(grad >= np.percentile(grad, percentile)).astype(int) for grad in layer_grads]
                        layer.set_masks(new_masks)
                    else:
                        layer_grads = [np.abs(grad.numpy()) for grad in grads[gradient_idx]]
                        new_masks = []
                        for grad in layer_grads:
                            new_mask = np.zeros(grad.shape)
                            used_weights = np.zeros(grad.shape[1:])
                            for task_idx in range(grad.shape[0]):
                                grad[task_idx][used_weights == 1] = 0
                                new_mask[task_idx] = (grad[task_idx] >= np.percentile(grad[task_idx], percentile)).astype(int)
                                used_weights += new_mask[task_idx]
                            new_masks.append(new_mask)
                        layer.set_masks(new_masks)
                    gradient_idx += 1
        # Work to identify the right weights if not exclusive
        else:
            gradient_idx = 0
            for layer in model.layers:
                if isinstance(layer, MASKING_LAYERS):
                    if not isinstance(layer, MULTI_MASKING_LAYERS):
                        layer_grads = [np.abs(grad.numpy()) for grad in grads[gradient_idx]]
                        new_masks = [(grad >= np.percentile(grad, percentile)).astype(int) for grad in layer_grads]
                        layer.set_masks(new_masks)
                    else:
                        layer_grads = [np.abs(grad.numpy()) for grad in grads[gradient_idx]]
                        new_masks = []
                        for grad in layer_grads:
                            new_mask = np.zeros(grad.shape)
                            for task_idx in range(grad.shape[0]):
                                new_mask[task_idx] = (grad[task_idx] >= np.percentile(grad[task_idx], percentile)).astype(int)
                            new_masks.append(new_mask)
                        layer.set_masks(new_masks)
                    gradient_idx += 1

    # Do this is method is "magnitude"
    elif method == 'magnitude':
        for layer in model.layers:
            if isinstance(layer, MASKING_LAYERS):
                if not isinstance(layer, MULTI_MASKING_LAYERS):
                    weights = [np.abs(weight.numpy()) for weight in layer.trainable_weights]
                    new_masks = [
                        (weight >= np.percentile(weight, percentile)).astype(int) for weight in weights
                    ]
                    layer.set_masks(new_masks)
                else:
                    weights = [np.abs(weight.numpy()) for weight in layer.trainable_weights]
                    if not exclusive:
                        new_masks = [np.zeros(weight.shape) for weight in weights]
                        for weight_idx in range(len(weights)):
                            for task_idx in range(weights[weight_idx].shape[0]):
                                new_masks[weight_idx][task_idx] = (weights[weight_idx][task_idx] >= np.percentile(weights[weight_idx][task_idx], percentile)).astype(int)
                    else:
                        new_masks = [np.zeros(weight.shape) for weight in weights]
                        for weight_idx in range(len(weights)):
                            for task_idx in range(weights[weight_idx].shape[0]):
                                exclusive_weight = weights[weight_idx][task_idx] * (1 - new_masks[weight_idx][:task_idx].sum(axis = 0))
                                new_masks[weight_idx][task_idx] = (exclusive_weight >= np.percentile(weights[weight_idx][task_idx], percentile)).astype(int)
                    layer.set_masks(new_masks)

    # Compile the model again so the effects take place
    model.compile()
    return model

def remove_layer_masks(model):
    """
    Convert a trained model from using Masking layers to using non-masking layers

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to be converted
    
    Returns
    -------
    new_model : TensorFlow Keras model
        The converted model
    """
    layer_mapping = {
        'MaskedConv2D' : 'Conv2D',
        'MaskedDense' : 'Dense',
        'MultiMaskedConv2D' : 'MultiConv2D',
        'MultiMaskedDense' : 'MultiDense'
    }
    
    config = model.get_config()
    for i in range(len(config['layers'])):
        if config['layers'][i]['class_name'] in layer_mapping.keys():
            config['layers'][i]['class_name'] = layer_mapping[
                config['layers'][i]['class_name']
            ]
            del config['layers'][i]['config']['mask_initializer']
    new_model = tf.keras.models.Model().from_config(
        config,
        custom_objects = get_custom_objects()
    )
    for i in range(len(model.layers)):
        if not isinstance(model.layers[i], MASKING_LAYERS):
            new_model.layers[i].set_weights(model.layers[i].get_weights())
        else:
            n_weights = len(new_model.layers[i].get_weights())
            new_model.layers[i].set_weights(model.layers[i].get_weights()[:n_weights])
    
    new_model.trainable = False
    new_model.compile()
    return new_model
