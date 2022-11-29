from beyondml.pt.layers import MaskedConv2D, MaskedConv3D, MaskedDense, MultiMaskedConv2D, MultiMaskedConv3D, MultiMaskedDense, MultitaskNormalization


def prune_model(model, percentile):
    """
    Prune a compatible model

    Parameters
    ----------
    model : PyTorch model
        A model that has been developed to have a `.layers` property containing layers to be pruned
    percentile : int
        An integer between 0 and 99 which corresponds to how much to prune the model

    Returns
    -------
    pruned_model : PyTorch model
        The pruned model

    Notes
    -----
    - The model input **must** have a `.layers` property to be able to function. Only layers within the
      `.layers` property which are recognized as prunable are pruned, via their own `.prune()` method
    - Also acts on the model in place, but returns the model for ease of use
    """

    compatible_layers = (MaskedConv2D, MaskedConv3D, MaskedDense,
                         MultiMaskedConv2D, MultiMaskedConv3D, MultiMaskedDense)

    try:
        for layer in model.layers:
            if isinstance(layer, compatible_layers):
                layer.prune(percentile)
    except AttributeError:
        raise AttributeError('Input model does not have a `.layers` attribute. Please make sure to add that attribute\
        to the model class in order to use this function')

    return model
