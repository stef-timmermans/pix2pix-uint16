"""Model entrypoints for the supported image-to-image trainers."""

from models.pix2pix_model import Pix2PixModel


MODEL_REGISTRY = {
    "pix2pix": Pix2PixModel,
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise NotImplementedError(
            f"Unsupported model '{model_name}'. Supported values: {supported}."
        )
    return MODEL_REGISTRY[model_name]


def get_option_setter(model_name: str):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = get_model_class(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option."""
    model = get_model_class(opt.model)
    instance = model(opt)
    print(f"model [{type(instance).__name__}] was created")
    return instance
