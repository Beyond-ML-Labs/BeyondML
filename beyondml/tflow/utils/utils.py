from beyondml.tflow.layers import MaskedDense, MaskedConv2D, MaskedConv3D, FilterLayer, SumLayer, SelectorLayer, MultiMaskedDense, MultiMaskedConv2D, MultiMaskedConv3D, MultiDense, MultiConv2D, MultiConv3D, MultiMaxPool2D, MultiMaxPool3D, SparseDense, SparseConv2D, SparseConv3D, SparseMultiDense, SparseMultiConv2D, SparseMultiConv3D, MultitaskNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import warnings

MASKING_LAYERS = (MaskedDense, MaskedConv2D, MaskedConv3D,
                  MultiMaskedDense, MultiMaskedConv2D, MultiMaskedConv3D)
MULTI_MASKING_LAYERS = (MultiMaskedDense, MultiMaskedConv2D, MultiMaskedConv3D)
NON_MASKING_LAYERS = (MultiDense, MultiConv2D, MultiConv3D)
SPARSE_LAYERS = (SparseDense, SparseConv2D, SparseConv3D,
                 SparseMultiDense, SparseMultiConv2D, SparseMultiConv3D)
CUSTOM_LAYERS = MASKING_LAYERS + NON_MASKING_LAYERS + SPARSE_LAYERS + \
    (FilterLayer, SumLayer, SelectorLayer, MultiMaxPool2D,
     MultiMaxPool3D, MultitaskNormalization)


class ActiveSparsification(Callback):
    """
    Keras-compatible callback object which enables active sparsification, allowing for increased sparsification as models
    train.
    """

    def __init__(
        self,
        performance_cutoff,
        performance_measure='auto',
        starting_sparsification=None,
        max_sparsification=99,
        sparsification_rate=1,
        sparsification_patience=10,
        stopping_delta=0.01,
        stopping_patience=5,
        restore_best_weights=True,
        verbose=1
    ):
        """
        Parameters
        ----------
        performance_cutoff : float
            The cutoff value that the performance measure must "beat" in order to iteratively sparsify
        performance_measure : str (default 'auto')
            The performance measure that is used in conjunction with `performance_cutoff`
        starting_sparsification : int or None (default None)
            The starting sparsification that the model has already been sparsified to. If `None`, then defaults to 0
        max_sparsification : int (default 99)
            The maximum sparsification allowed
        sparsification_rate : int (default 1)
            The increase in sparsification that occurs when model performance beats the performance cutoff
        sparsification_patience : int (default 10)
            The number of epochs the model is allowed to train for without beating the performance measure before stopping sparsification
        stopping_delta : float (default 0.01)
            The performance improvement that must be seen when pruning has stopped and early stopping is being considered
        stopping_patience : int (default 5)
            The number of epochs the model is allowed to train for without `stopping_delta` improvement before stopping training
        restore_best_weights : bool (default True)
            Whether to restore model best weights after training
        verbose : int or bool (default 1)
            Verbosity level for logging.

        Notes
        -----
        - If `performance_measure` is 'auto', defaults to the following measures, in order: 'val_accuracy', 'val_loss', 'accuracy', 'loss'
        - If `performance_measure` defaults to any accuracy value, then `performance_cutoff` represents the minimum value that must be
          beaten.  If `performance_measure` defaults to any loss value, then `performance_cutoff` represents the maximum value that must
          be beaten
        """
        super(Callback, self).__init__()
        self.performance_cutoff = performance_cutoff
        self.performance_measure = performance_measure
        self.starting_sparsification = starting_sparsification
        self.max_sparsification = max_sparsification
        self.sparsification_rate = sparsification_rate
        self.sparsification_patience = sparsification_patience
        self.stopping_delta = stopping_delta
        self.stopping_patience = stopping_patience
        self.restore_best_weights = restore_best_weights
        self.verbose = int(verbose)

    def on_train_begin(self, logs=None):
        self.prune_wait = 0
        self.stop_wait = 0

        self.best_weights = self.model.get_weights()
        self.best = None
        self.pruning = True
        self.sparsification = self.starting_sparsification if self.starting_sparsification is not None else 0

    def on_epoch_end(self, epoch, logs=None):
        if self.performance_measure == 'auto':
            if 'val_accuracy' in logs.keys():
                self.performance_measure = 'val_accuracy'
            elif 'val_loss' in logs.keys():
                self.performance_measure = 'val_loss'
            elif 'accuracy' in logs.keys():
                self.performance_measure = 'accuracy'
            else:
                self.performance_measure = 'loss'
            if self.verbose:
                print(f'Performance measure set to {self.performance_measure}')

        performance = logs[self.performance_measure]
        if self.best is None:
            self.best = performance

        if 'accuracy' in self.performance_measure:
            if self.pruning:
                if performance >= self.performance_cutoff:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    if self.sparsification + self.sparsification_rate > self.max_sparsification:
                        print(
                            'Model cannot be sparsified further due to max sparsification parameter')
                        self.pruning = False
                    else:
                        self._sparsify_model(
                            self.sparsification + self.sparsification_rate)
                        self.sparsification = self.sparsification + self.sparsification_rate
                        self.prune_wait = 0
                        if self.verbose:
                            print(
                                f'Model performance reached {round(performance, 2)}, sparsifying to {self.sparsification}')
                else:
                    self.prune_wait += 1
                    if self.verbose:
                        print(
                            f'Model performance has not reached pruning threshold for {self.prune_wait} epoch(s)')
                    if self.prune_wait >= self.sparsification_patience:
                        self.pruning = False
                        self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(
                                f'Model performance has not reached pruning threshold for {self.prune_wait} epochs, reverting to {self.sparsification - self.sparsification_rate} sparsification and beginning early stopping')
            else:
                if performance >= self.best + self.stopping_delta:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    self.stop_wait = 0
                    if self.verbose:
                        print(
                            f'Model performance improved to {round(self.best, 2)}')
                else:
                    self.stop_wait += 1
                    if self.verbose:
                        print(
                            f'Early stopping performance has not met threshold for {self.stop_wait} epochs')
                    if self.stop_wait >= self.stopping_patience:
                        if self.restore_best_weights:
                            self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(
                                'Model performance has not met early stopping criteria. Stopping training')
                        self.model.stop_training = True

        else:
            if self.pruning:
                if performance <= self.performance_cutoff:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    if self.sparsification + self.sparsification_rate > self.max_sparsification:
                        print(
                            'Model cannot be sparsified further due to max sparsification parameter')
                        self.pruning = False
                    else:
                        self._sparsify_model(
                            self.sparsification + self.sparsification_rate)
                        self.sparsification = self.sparsification + self.sparsification_rate
                        self.prune_wait = 0
                        if self.verbose:
                            print(
                                f'Model performance reached {round(performance, 2)}, sparsifying to {self.sparsification}')
                else:
                    self.prune_wait += 1
                    if self.verbose:
                        print(
                            f'Model performance has not reached pruning threshold for {self.prune_wait} epoch(s)')
                    if self.prune_wait >= self.sparsification_patience:
                        self.pruning = False
                        self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(
                                f'Model performance has not reached pruning threshold for {self.prune_wait} epochs, reverting to {self.sparsification - self.sparsification_rate} sparsification and beginning early stopping')
            else:
                if performance <= self.best - self.stopping_delta:
                    self.best_weights = self.model.get_weights()
                    self.best = performance
                    self.stop_wait = 0
                    if self.verbose:
                        print(
                            f'Model performance improved to {round(self.best, 2)}')
                else:
                    self.stop_wait += 1
                    if self.verbose:
                        print(
                            f'Early stopping performance has not met threshold for {self.stop_wait} epochs')
                    if self.stop_wait >= self.stopping_patience:
                        if self.restore_best_weights:
                            self.model.set_weights(self.best_weights)
                        if self.verbose:
                            print(
                                'Model performance has not met early stopping criteria. Stopping training')
                        self.model.stop_training = True

    def _sparsify_model(self, percentage):
        """Function to sparsify the model"""
        try:
            new_model = tf.keras.models.Model.from_config(
                self.model.get_config(), custom_objects=get_custom_objects())
        except Exception:
            new_model = tf.keras.models.Sequential.from_config(
                self.model.get_config(), custom_objects=get_custom_objects())
        new_model.set_weights(self.model.get_weights())
        self.model.set_weights(
            mask_model(
                new_model,
                percentage,
                method='magnitude'
            ).get_weights()
        )


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
            raise ValueError(
                'Error in output shapes. If any tasks have a single output, please reshape the value using the `.reshape(-1, 1)` method')
    elif not len(y.shape) > 1:
        raise ValueError(
            'Error in output shapes. If your task has a single output, please reshape the value using the `.reshape(-1, 1)` method')

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
        losses = [tf.keras.losses.get(loss) if not callable(
            loss) else loss for loss in losses]

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
            ['MaskedDense', 'MaskedConv2D', 'MaskedConv3D', 'MultiMaskedDense', 'MultiMaskedConv2D', 'MultiMaskedConv3D', 'MultiDense',
                'MultiConv2D', 'MultiConv3D', 'SparseDense', 'SparseConv2D', 'SparseConv3D', 'SparseMultiDense', 'SparseMultiConv2D', 'SparseMultiConv3D',
                'FilterLayer', 'SumLayer', 'SelectorLayer', 'MultiMaxPool2D', 'MultiMaxPool3D'],
            CUSTOM_LAYERS
        )
    )


def mask_model(
        model,
        percentile,
        method='gradients',
        exclusive=True,
        x=None,
        y=None
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
        raise ValueError(
            f"method must be one of 'gradients', 'magnitude', got {method}")

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
                if isinstance(layer, tf.keras.models.Model):
                    warnings.warn(
                        'mask_model does not effectively support models with models as layers if method is "gradients". Please set method to "magnitude"', RuntimeWarning)
                if isinstance(layer, MASKING_LAYERS):
                    if not isinstance(layer, MULTI_MASKING_LAYERS):
                        layer_grads = [np.abs(grad)
                                       for grad in grads[gradient_idx]]
                        new_masks = [(grad >= np.percentile(grad, percentile)).astype(
                            int) for grad in layer_grads]
                        layer.set_masks(new_masks)
                    else:
                        layer_grads = [np.abs(grad.numpy())
                                       for grad in grads[gradient_idx]]
                        new_masks = []
                        for grad in layer_grads:
                            new_mask = np.zeros(grad.shape)
                            used_weights = np.zeros(grad.shape[1:])
                            for task_idx in range(grad.shape[0]):
                                grad[task_idx][used_weights == 1] = 0
                                new_mask[task_idx] = (grad[task_idx] >= np.percentile(
                                    grad[task_idx], percentile)).astype(int)
                                used_weights += new_mask[task_idx]
                            new_masks.append(new_mask)
                        layer.set_masks(new_masks)
                    gradient_idx += 1
        # Work to identify the right weights if not exclusive
        else:
            gradient_idx = 0
            for layer in model.layers:
                if isinstance(layer, tf.keras.models.Model):
                    warnings.warn(
                        'mask_model does not effectively support models with models as layers if method is "gradients". Please set method to "magnitude"', RuntimeWarning)
                if isinstance(layer, MASKING_LAYERS):
                    if not isinstance(layer, MULTI_MASKING_LAYERS):
                        layer_grads = [np.abs(grad.numpy())
                                       for grad in grads[gradient_idx]]
                        new_masks = [(grad >= np.percentile(grad, percentile)).astype(
                            int) for grad in layer_grads]
                        layer.set_masks(new_masks)
                    else:
                        layer_grads = [np.abs(grad.numpy())
                                       for grad in grads[gradient_idx]]
                        new_masks = []
                        for grad in layer_grads:
                            new_mask = np.zeros(grad.shape)
                            for task_idx in range(grad.shape[0]):
                                new_mask[task_idx] = (grad[task_idx] >= np.percentile(
                                    grad[task_idx], percentile)).astype(int)
                            new_masks.append(new_mask)
                        layer.set_masks(new_masks)
                    gradient_idx += 1

    # Do this is method is "magnitude"
    elif method == 'magnitude':
        for layer in model.layers:
            if isinstance(layer, MASKING_LAYERS):
                if not isinstance(layer, MULTI_MASKING_LAYERS):
                    weights = [np.abs(weight.numpy())
                               for weight in layer.trainable_weights]
                    new_masks = [
                        (weight >= np.percentile(weight, percentile)).astype(int) for weight in weights
                    ]
                    layer.set_masks(new_masks)
                else:
                    weights = [np.abs(weight.numpy())
                               for weight in layer.trainable_weights]
                    if not exclusive:
                        new_masks = [np.zeros(weight.shape)
                                     for weight in weights]
                        for weight_idx in range(len(weights)):
                            for task_idx in range(weights[weight_idx].shape[0]):
                                new_masks[weight_idx][task_idx] = (weights[weight_idx][task_idx] >= np.percentile(
                                    weights[weight_idx][task_idx], percentile)).astype(int)
                    else:
                        new_masks = [np.zeros(weight.shape)
                                     for weight in weights]
                        for weight_idx in range(len(weights)):
                            for task_idx in range(weights[weight_idx].shape[0]):
                                exclusive_weight = weights[weight_idx][task_idx] * (
                                    1 - new_masks[weight_idx][:task_idx].sum(axis=0))
                                new_masks[weight_idx][task_idx] = (exclusive_weight >= np.percentile(
                                    weights[weight_idx][task_idx], percentile)).astype(int)
                    layer.set_masks(new_masks)
            elif isinstance(layer, tf.keras.models.Model):
                mask_model(
                    layer,
                    percentile,
                    method,
                    exclusive
                )

    # Compile the model again so the effects take place
    model.compile()
    return model


def _replace_config(config):
    """
    Replace the model config to remove masking layers
    """

    layer_mapping = {
        'MaskedConv2D': 'Conv2D',
        'MaskedConv3D': 'Conv3D',
        'MaskedDense': 'Dense',
        'MultiMaskedConv2D': 'MultiConv2D',
        'MultiMaskedConv3D': 'MultiConv3D',
        'MultiMaskedDense': 'MultiDense'
    }
    model_classes = ('Functional', 'Sequential')

    for i in range(len(config['layers'])):
        if config['layers'][i]['class_name'] in layer_mapping.keys():
            config['layers'][i]['class_name'] = layer_mapping[
                config['layers'][i]['class_name']
            ]
            del config['layers'][i]['config']['mask_initializer']
        elif config['layers'][i]['class_name'] in model_classes:
            config['layers'][i]['config'] = _replace_config(
                config['layers'][i]['config'])
    return config


def _create_masking_config(config):
    """
    Replace the model config to add masking layers
    """

    layer_mapping = {
        'Conv2D': 'MaskedConv2D',
        'Conv3D': 'MaskedConv3D',
        'Dense': 'MaskedDense',
        'MultiConv2D': 'MultiMaskedConv2D',
        'MultiConv3D': 'MultiMaskedConv3D',
        'MultiDense': 'MultiMaskedDense'
    }
    model_classes = ('Functional', 'Sequential')

    for i in range(len(config['layers'])):
        if config['layers'][i]['class_name'] in layer_mapping.keys():
            config['layers'][i]['class_name'] = layer_mapping[
                config['layers'][i]['class_name']
            ]
            config['layers'][i]['config']['mask_initializer'] = tf.keras.initializers.serialize(
                tf.keras.initializers.get('ones')
            )
        elif config['layers'][i]['class_name'] in model_classes:
            config['layers'][i]['config'] = _create_masking_config(
                config['layers'][i]['config'])
    return config


def _quantize_model_config(config, dtype='float16'):
    """
    Change the dtype of the model
    """

    model_classes = ('Functional', 'Sequential')

    new_config = config.copy()
    for i in range(len(new_config['layers'])):
        if new_config['layers'][i]['class_name'] in model_classes:
            new_config['layers'][i] = _quantize_model_config(
                new_config['layers'][i]['config'], dtype)
        else:
            new_config['layers'][i]['config']['dtype'] = dtype
    return new_config


def _replace_weights(new_model, old_model):
    """
    Replace the weights of a newly created model with the weights (sans masks) of an old model
    """

    for i in range(len(new_model.layers)):
        # Recursion in case the model contains other models
        if isinstance(new_model.layers[i], tf.keras.models.Model):
            _replace_weights(new_model.layers[i], old_model.layers[i])

        # If not masking layers, simply replace weights
        elif not isinstance(old_model.layers[i], MASKING_LAYERS):
            new_model.layers[i].set_weights(old_model.layers[i].get_weights())

        # If masking layers, replace only the required weights
        else:
            n_weights = len(new_model.layers[i].get_weights())
            new_model.layers[i].set_weights(
                old_model.layers[i].get_weights()[:n_weights])

    # Compile and return the model
    new_model.compile()
    return new_model


def _replace_masking_weights(new_model, old_model):
    """
    Replace the weights of a newly created model with the weights (adding masks) of an old model
    """

    for i in range(len(new_model.layers)):
        # Recursion in case the model contains other models
        if isinstance(new_model.layers[i], tf.keras.models.Model):
            _replace_masking_weights(new_model.layers[i], old_model.layers[i])

        # If not masking layers, simply replace weights
        elif not isinstance(new_model.layers[i], MASKING_LAYERS):
            new_model.layers[i].set_weights(old_model.layers[i].get_weights())

        # If masking layers, replace the weights and have all ones as the masks
        else:
            n_weights = len(old_model.layers[i].get_weights())
            weights = old_model.layers[i].get_weights()
            weights.extend(new_model.layers[i].get_weights()[n_weights:])
            new_model.layers[i].set_weights(weights)

    # Compile and return the model
    new_model.compile()
    return new_model


def remove_layer_masks(model, additional_custom_objects=None):
    """
    Convert a trained model from using Masking layers to using non-masking layers

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to be converted
    additional_custom_objects : dict or None (default None)
        Additional custom layers to use

    Returns
    -------
    new_model : TensorFlow Keras model
        The converted model
    """

    custom_objects = get_custom_objects()
    if additional_custom_objects is not None:
        custom_objects.update(additional_custom_objects)

    # Replace the config of the model
    config = model.get_config()
    new_config = _replace_config(config)

    # Create the new model
    try:
        new_model = tf.keras.models.Model().from_config(
            new_config,
            custom_objects=custom_objects
        )
    except Exception:
        new_model = tf.keras.models.Sequential().from_config(
            new_config,
            custom_objects=custom_objects
        )

    # Replace the weights of the new model to be equivalent to the old model
    new_model = _replace_weights(new_model, model)

    # Make the new model not trainable and compile the model for good measure
    new_model.trainable = False
    new_model.compile()
    return new_model


def add_layer_masks(model, additional_custom_objects=None):
    """
    Convert a trained model from one that does not have masking weights to one that does have
    masking weights

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to be converted
    additional_custom_objects : dict or None (default None)
        Additional custom layers to use

    Returns
    -------
    new_model : TensorFlow Keras model
        The converted model
    """

    custom_objects = get_custom_objects()
    if additional_custom_objects is not None:
        custom_objects.update(additional_custom_objects)

    # Replace the config of the model
    config = model.get_config()
    new_config = _create_masking_config(config)

    # Create the new model
    try:
        new_model = tf.keras.models.Model().from_config(
            new_config,
            custom_objects=custom_objects
        )
    except Exception:
        new_model = tf.keras.models.Sequential().from_config(
            new_config,
            custom_objects=custom_objects
        )

    # Replace the weights of the new model
    new_model = _replace_masking_weights(new_model, model)

    # Compile and return the model
    new_model.compile()
    return new_model


def quantize_model(model, dtype='float16', additional_custom_objects=None):
    """
    Apply model quantization

    Parameters
    ----------
    model : TensorFlow Keras Model
        The model to quantize
    dtype : str or TensorFlow datatype (default 'float16')
        The datatype to quantize to
    additional_custom_objects : None or dict (default None)
        Additional custom  objects to use to instantiate the model

    Returns
    -------
    new_model : TensorFlow Keras Model
        The quantized model
    """

    # Grab the configuration from the original model
    model_config = model.get_config()

    # Grab the weights from the original model as well
    weights = model.get_weights()

    # Change the weights to have the new datatype
    new_weights = [
        np.array(w, dtype=dtype) for w in weights
    ]

    # Change the config to get the quantized configuration
    new_config = _quantize_model_config(model_config, dtype)

    # Instantiate the new model from the new config
    custom_objects = get_custom_objects()
    if additional_custom_objects is not None:
        custom_objects.update(additional_custom_objects)
    try:
        new_model = tf.keras.models.Model.from_config(
            new_config, custom_objects=custom_objects)
    except Exception:
        new_model = tf.keras.models.Sequential.from_config(
            new_config, custom_objects=custom_objects)

    # Set the weights of the new model
    new_model.set_weights(new_weights)
    return new_model


def _get_masking_weights(model):
    """
    Get the masking weights of a model

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to get the masking weights of

    Returns
    -------
    weights : list of TensorFlow tensors
        The requested weights
    """
    return [
        layer.weights for layer in model.layers if isinstance(layer, MASKING_LAYERS)
    ]


def get_task_masking_gradients(
    model,
    task_num
):
    """
    Get the gradients of masking weights within a model

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to retrieve the gradients of

    Notes
    -----
    - This function should only be run *before* the model has been trained
        or used to predict.  There is an unknown bug related to TensorFlow which
        is leading to incorrect results after initial training
    - When running this function, randomized input and output data is sent
        through the model to retrieve gradients respective to each task. If
        the model is compiled using `sparse_categorical_crossentropy' loss,
        this will break this function's functionality. As a result, please
        use `categorical_crossentropy` (or even better, `mse`) before running this function. After
        retrieving gradients, the model can be recompiled with whatever parameters are desired.

    Returns
    -------
    gradients : list of TensorFlow tensors
        The gradients of the masking weights of the model
    """
    # Figure out the number of tasks
    output_shapes = model.output_shape
    if isinstance(output_shapes, list):
        num_tasks = len(output_shapes)
    else:
        num_tasks = 1

    # Get the loss weights
    if num_tasks > 1:
        loss_weights = [0] * num_tasks
        loss_weights[task_num] = 1

    # Get the masking weights
    masking_weights = _get_masking_weights(model)

    # Configure inputs
    inputs = []
    input_shapes = model.input_shape
    if isinstance(input_shapes, list):
        for shape in input_shapes:
            new_shape = list(shape)
            for i in range(len(new_shape)):
                if new_shape[i] is None:
                    new_shape[i] = 1
            inputs.append(np.random.random(new_shape))
    else:
        new_shape = list(input_shapes)
        for i in range(len(new_shape)):
            if new_shape[i] is None:
                new_shape[i] = 1
        inputs.append(np.random.random(new_shape))

    # Configure outputs
    outputs = []
    output_shapes = model.output_shape
    if isinstance(output_shapes, list):
        for shape in output_shapes:
            new_shape = list(shape)
            for i in range(len(new_shape)):
                if new_shape[i] is None:
                    new_shape[i] = 1
            outputs.append(np.random.random(new_shape))
    else:
        new_shape = list(output_shapes)
        for i in range(len(new_shape)):
            if new_shape[i] is None:
                new_shape[i] = 1
        outputs.append(np.random.random(new_shape))

    # Configure the losses
    losses = model.loss
    if not isinstance(losses, list):
        losses = [losses] * num_tasks
    losses = [
        tf.keras.losses.get(loss) for loss in losses
    ]

    # Get the gradients of the weights wrt the task
    with tf.GradientTape() as tape:
        raw_preds = model(inputs)
        loss_values = [losses[i](outputs[i], raw_preds[i]) * loss_weights[i]
                       for i in range(len(losses))]
        gradients = tape.gradient(loss_values, masking_weights)

    return gradients


def mask_task_weights(
    model,
    task_masking_gradients,
    percentile,
    respect_previous_tasks=True
):
    """
    Parameters
    ----------
    model : TensorFlow Keras model
        The model to be masked
    task_masking_gradients : list of TensorFlow tensors
        The gradients for the specific task requested
    percentile : int
        The percentile to mask/prune
    respect_previous_tasks : bool (default True)
        Whether to respect the weights used for previous tasks and not use them
        for subsequent tasks

    Returns
    -------
    masked_model : TensorFlow Keras model
        The masked model
    """

    # Get the actual weights to be able to set them
    masking_weights = [
        layer.get_weights() for layer in model.layers if isinstance(layer, MASKING_LAYERS)
    ]

    # Iterate through each of the layers of the model, keeping track of the index of which masking layer has been achieved
    masking_idx = 0
    for layer in model.layers:
        if isinstance(layer, MASKING_LAYERS):

            # Set the new masks to be masked
            new_masks = []

            # Different procedures if multi masking layer vs single masking layer
            if isinstance(layer, MULTI_MASKING_LAYERS):

                # Check for all of the weights in the list of weights (corresponding to gradients)
                for weight_num in range(len(task_masking_gradients[masking_idx])):

                    # Set the weight and gradient values so it's easier to follow
                    weight = masking_weights[masking_idx][weight_num]
                    gradient = task_masking_gradients[masking_idx][weight_num]

                    # If gradient is None, then the value is a mask
                    task_idx_num = None
                    if gradient is not None:

                        # Figure out which task index is the right one
                        for task_idx in range(gradient.shape[0]):
                            if not (gradient[task_idx].numpy() == 0).all():
                                task_idx_num = task_idx

                        if task_idx_num is not None:
                            # Get the new weight for that task only
                            task_weight = np.abs(weight[task_idx_num])

                            # Enforce respecting previous-task weights
                            if respect_previous_tasks and task_idx_num > 0:
                                task_weight[(weight[:task_idx_num] != 0).astype(
                                    int).sum(axis=0).astype(bool)] = 0

                            # Get the new mask
                            weight_mask = (task_weight >= np.percentile(
                                task_weight, percentile))

                            # Find the existing mask and set the value of only the task-specific part
                            layer_mask = masking_weights[masking_idx][weight_num + int(
                                len(masking_weights[masking_idx]) / 2)]
                            layer_mask[task_idx_num] = weight_mask

                            # Append the new mask
                            new_masks.append(layer_mask)

            # If the layer is a single masking layer
            else:
                for weight_num in range(len(task_masking_gradients[masking_idx])):

                    # Assign the weight and the gradient for this specific layer, for sanity
                    weight = masking_weights[masking_idx][weight_num]
                    gradient = task_masking_gradients[masking_idx][weight_num]

                    # If gradient is None, then the weight is a mask
                    if gradient is not None:

                        # Only proceed if the gradient exists
                        if not (gradient.numpy() == 0).all():
                            weight = np.abs(weight)
                            weight_mask = (
                                weight >= np.percentile(weight, percentile))
                            new_masks.append(weight_mask)

            # If new masks have been identified (it's possible that this did not occur), set the new masks for that layer
            if new_masks != []:
                layer.set_masks(new_masks)

            # Lastly, increase the masking index by 1
            masking_idx += 1

    # Compile the model again and return it
    model.compile()
    return model


def train_model_iteratively(
    model,
    task_gradients,
    train_x,
    train_y,
    validation_split,
    delta,
    batch_size,
    losses,
    optimizer='adam',
    metrics=None,
    starting_pruning=0,
    pruning_rate=10,
    patience=5,
    max_epochs=100
):
    """
    Train a model iteratively on each task, first obtaining
    baseline performance on each task and then iteratively
    training and pruning each task as far back as possible while
    maintaining acceptable performance on each task

    Parameters
    ----------
    model : TensorFlow Keras model
        The model to be trained
    task_gradients : list of TensorFlow tensors
        Gradients for each task, output from the `get_task_masking_gradients` function
    train_x : list of numpy arrays, TensorFlow Datasets, or other
              data types models can train with
        The input data to use to train on
    train_y : list of numpy arrays, TensorFlow Datasets, or other
              data types model can train with
        The output data to use to train on
    validation_split : float, or list of float
        The proportion of data to use for validation
    delta : float
        The tolerance between validation losses to be considered "acceptable"
        performance to continue
    batch_size : int
        The batch size to train with
    losses : str, list, or Keras loss function
        The loss or losses to use when training
    optimizer : str, list, or Keras optimizer
        The optimizer to use when training (default 'adam')
    starting_pruning : int or list of int (default 0)
        The starting pruning rate to use for each task
    pruning_rate : int or list of int (default [10, 5, 2, 1])
        The pruning rate to use
    patience : int (default 5)
        The patience for number of epochs to wait for performance to improve sufficiently
    max_epochs : int or list of int (default 100)
        The maximum number of epochs to use for training each task
    """

    # Get some information about the training procedure, including the number of tasks
    # and the gradients for each task
    num_tasks = len(train_x)

    # Keep track of the amount of the model currently used
    amount_used = 100

    # Start the training iterations
    for task_num in range(num_tasks):

        print(f'Training task {task_num}')

        # Get the starting task pruning rate for the current task
        if isinstance(starting_pruning, int):
            task_start_pruning = starting_pruning
        else:
            task_start_pruning = starting_pruning[task_num]

        # Get the current pruning rate
        if isinstance(pruning_rate, int):
            current_pruning_rate = pruning_rate
        else:
            current_pruning_rate = pruning_rate[task_num]

        # Configure the current validation split
        if isinstance(validation_split, float):
            current_validation_split = validation_split
        else:
            current_validation_split = validation_split[task_num]

        # Configure the loss weights
        loss_weights = [0] * num_tasks
        loss_weights[task_num] = 1

        # Configure the epochs
        if isinstance(max_epochs, int):
            current_epochs = max_epochs
        else:
            current_epochs = max_epochs[task_num]

        # Compile the model
        model.compile(
            loss=losses,
            optimizer=optimizer,
            loss_weights=loss_weights,
            metrics=metrics
        )

        # Train the model initially
        callback = tf.keras.callbacks.EarlyStopping(
            min_delta=delta,
            patience=patience,
            restore_best_weights=True
        )
        history = model.fit(
            train_x[task_num],
            train_y[task_num],
            batch_size=batch_size,
            epochs=current_epochs,
            validation_split=current_validation_split,
            callbacks=[callback],
            verbose=2
        )

        # Retrieve the validation loss and current best weights
        best_loss = min(history.history['val_loss'])
        best_weights = model.get_weights()

        # Training loop for the task at hand
        current_wait = 0
        if task_num == 0:
            current_prune = task_start_pruning
        else:
            current_prune = max(task_start_pruning, amount_used)
        keep_training = True
        just_started = True

        # If pruning needs to occur, do it
        if current_prune != 0:

            print(f'Pruning task to {current_prune}')

            model = mask_task_weights(
                model,
                task_gradients[task_num],
                current_prune
            )

        # keep_training indicates that training is to occur
        while keep_training:

            # First prune the model to the next pruning rate
            if current_prune + current_pruning_rate < 100:

                # Check if the training just started
                if not just_started:
                    # Increase the pruning rate
                    current_prune += current_pruning_rate
                    model = mask_task_weights(
                        model,
                        task_gradients[task_num],
                        current_prune
                    )

                    print(f'Pruning task to {current_prune}')

                else:
                    just_started = False

                # Recompile the model
                model.compile(
                    loss=losses,
                    optimizer=optimizer,
                    loss_weights=loss_weights,
                    metrics=metrics
                )

                # Train the model with the new pruning rate
                while current_wait < patience:

                    # Fit the model for a single epoch
                    history = model.fit(
                        train_x[task_num],
                        train_y[task_num],
                        batch_size=batch_size,
                        validation_split=current_validation_split,
                        verbose=2
                    )

                    # Get the new loss
                    loss = history.history['val_loss'][-1]

                    # If loss is within acceptable range, grab the best weights and
                    # reassign the best loss. Otherwise, increase current wait
                    if loss < best_loss + delta:
                        best_weights = model.get_weights()
                        best_loss = loss
                        break
                    else:
                        current_wait += 1

                # If pruning was not successful, restore best pruning rate
                if current_wait == patience or current_prune + current_pruning_rate >= 100:
                    keep_training = False

            else:
                keep_training = False

        # Record how much of the model has been used
        if task_num == 0:
            amount_used -= current_prune
        else:
            amount_used += 100 - current_prune

        # Now that current wait has been reached, restore best weights
        model.set_weights(best_weights)

        # Recompile the model
        model.compile(
            loss=losses,
            optimizer=optimizer,
            loss_weights=loss_weights,
            metrics=metrics
        )

        # Fit using the new best weights
        model.fit(
            train_x[task_num],
            train_y[task_num],
            batch_size=batch_size,
            epochs=current_epochs,
            validation_split=current_validation_split,
            callbacks=[callback],
            verbose=2
        )

    return model


def train_model(
    model,
    train_x,
    train_y,
    loss,
    metrics,
    optimizer,
    cutoff,
    batch_size=32,
    epochs=100,
    starting_sparsification=0,
    max_sparsification=99,
    sparsification_rate=5,
    sparsification_patience=10,
    stopping_patience=5
):
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    callback = ActiveSparsification(
        cutoff,
        starting_sparsification=starting_sparsification,
        max_sparsification=max_sparsification,
        sparsification_rate=sparsification_rate,
        sparsification_patience=sparsification_patience,
        stopping_patience=stopping_patience
    )

    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callback]
    )

    return model
