import timm


def create_timm_model(model_name, num_classes, use_pretrained=True, in_chans=3):
    """
    Create a model from the TIMM library with the specified parameters.

    Args:
        model_name (str): The name of the TIMM model to be created.
        num_classes (int): The number of output classes for the model.
        use_pretrained (bool): Flag indicating whether to use pretrained weights.
        in_chans (int): The number of input channels for the model.

    Returns:
        A model object created using the TIMM library.

    Raises:
        ValueError: If the specified model is invalid or does not have the requested pretrained weights.
    """
    available_models = timm.list_models()
    available_pretrained_models = timm.list_models(pretrained=True)

    # Check if the model is in the list of all models or pretrained models
    if use_pretrained:
        if model_name not in available_pretrained_models:
            related_pretrained = timm.list_models(f"*{model_name}*", pretrained=True)
            raise ValueError(
                f"No pretrained weights available for '{model_name}'. "
                f"Consider these pretrained options: {related_pretrained}"
            )
        print(f"Using pretrained model '{model_name}'.")
    else:
        base_model_name = model_name.split(".")[0]
        if model_name not in available_models:
            suggested_models = timm.list_models(f"*{base_model_name}*")
            if model_name in available_pretrained_models:
                raise ValueError(
                    f"Model '{model_name}' has pretrained weights. "
                    "Enable 'use_pretrained=True' to use them."
                )
            else:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Suggested model names: {suggested_models}"
                )
        else:
            suggested_pretrained = timm.list_models(
                f"*{base_model_name}*", pretrained=True
            )
            print(f"Using '{model_name}' without pretrained weights.")
            print(
                f"Consider a pretrained model for improved performance. "
                f"Pretrained options: {suggested_pretrained}"
            )

    # Create and return the model
    return timm.create_model(
        model_name,
        num_classes=num_classes,
        pretrained=use_pretrained,
        in_chans=in_chans,
    )
