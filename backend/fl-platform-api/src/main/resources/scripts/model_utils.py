import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

# --- THIS IS THE SIMPLIFIED IMPORT ---
# Because the wrapper script changes the directory, we can now use a simple relative import.
from architecture.cnn.net import Net as CnnNet

# --- The single source of truth for creating model architectures ---
def get_model(model_type: str, model_name: str, device: str):
    """Returns a model instance based on the architecture and name."""
    print(f"Loading model: {model_type} / {model_name}")
    model_type = model_type.upper()
    model_name = model_name.lower()

    if model_type == 'CNN':
        # Add elif branches here for 'resnet', 'vgg', etc. in the future
        return CnnNet().to(device)

    elif model_type == 'TRANSFORMER':
        if model_name == 'opt-125m':
            return AutoModelForSequenceClassification.from_pretrained(
                "facebook/opt-125m", num_labels=2, use_safetensors=True
            ).to(device)
        else:
            raise ValueError(f"Unsupported Transformer model: {model_name}")

    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")