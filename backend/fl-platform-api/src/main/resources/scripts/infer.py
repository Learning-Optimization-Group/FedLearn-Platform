"""
infer.py — single-input inference on a trained FedLearn model.

Invoked by the backend (InferenceService) via run_infer.sh. Reconstructs the
*same* architecture used during training, loads the aggregated weights from the
project's .npz file, runs one forward pass, and writes the result as JSON to the
path given by --out.

Contract (deliberate): the result is written to the --out FILE, never to stdout.
torch/CUDA banners and wrapper log lines pollute stdout, so the Java side reads
the out-file and treats stdout purely as diagnostic logging.

Supported model types (v1):
  CNN  — CnnNet (CIFAR-10), input 3x32x32 image, 10 classes.
  MLP  — ECGModel(140, 64, 2), input a 140-float vector, 2 classes.
  TRANSFORMER — not supported for interactive inference yet (returns ok=false).

Input file (JSON, written by the backend), one of:
  {"kind": "image",  "imagePath": "/abs/path/to/image"}
  {"kind": "vector", "values": [<floats>]}
"""

import argparse
import json
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F


class InputError(ValueError):
    """A problem with the caller-supplied payload (bad image, wrong vector length).

    Distinguished from internal faults (model load, arch import, torch errors) so
    the backend can map it to HTTP 400 instead of 502 and surface the message.
    """


# Largest decoded image we will accept, in pixels. Defuses decompression bombs
# (a tiny file that expands to a huge bitmap). We downscale to 32x32 anyway.
MAX_IMAGE_PIXELS = 50_000_000


# CIFAR-10 class order (torchvision's canonical ordering).
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
ECG_CLASSES = ["Normal", "Abnormal"]

ECG_INPUT_DIM = 140
ECG_HIDDEN_DIM = 64
ECG_NUM_CLASSES = 2


def log(msg: str) -> None:
    """Diagnostic logging to stdout — never the result channel."""
    print(f"[infer] {msg}", flush=True)


def build_model(model_type: str, model_name: str):
    """Reconstruct the architecture exactly as training did.

    Imports the architecture modules directly (not models.py/init_model.py) so we
    don't drag in the heavy `transformers` import for CNN/MLP inference.
    """
    mt = model_type.upper()
    if mt == "CNN":
        from architecture.cnn.net import Net  # 3x32x32 -> 10
        return Net(), CIFAR10_CLASSES, "image", None
    if mt == "PNEUMONIA_CNN":
        import recipes
        recipe = recipes.get_recipe("PNEUMONIA_CNN")  # 1x224x224 grayscale -> [NORMAL, PNEUMONIA]
        return recipe.build_model("cpu"), recipe.classes, "image", recipe.input_transform()
    if mt == "MLP":
        from models.ecg_mlp import ECGModel
        return (
            ECGModel(input_dim=ECG_INPUT_DIM, hidden_dim=ECG_HIDDEN_DIM, num_classes=ECG_NUM_CLASSES),
            ECG_CLASSES,
            "vector",
            None,
        )
    if mt == "TRANSFORMER":
        raise ValueError("Transformer (text) models are not supported for interactive inference yet.")
    raise ValueError(f"Unsupported model type: {model_type}")


def load_weights(net, model_path: str) -> None:
    """Load the aggregated state_dict from the .npz (keys use __DOT__ for '.')."""
    state = {}
    with np.load(model_path, allow_pickle=False) as npz:
        for key in npz.files:
            value = npz[key]
            if isinstance(value, np.ndarray):
                state[key.replace("__DOT__", ".")] = torch.from_numpy(value)
    if not state:
        raise ValueError("No parameters found in model file.")
    net.load_state_dict(state, strict=True)


def build_image_tensor(image_path: str, transform=None) -> torch.Tensor:
    """Decode an image to the model's expected input tensor.

    Default (CnnNet): 3x32x32 normalized to [-1, 1]. If `transform` is provided
    (e.g. the PneumoniaCNN 1x224x224 grayscale transform), it is used instead —
    this MUST mirror the training transform for that model.
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # Hard cap on decoded resolution before we touch pixel data — a malformed or
    # adversarial image is caller input, so failures here are InputError (→ 400).
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        img = Image.open(image_path)
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            raise InputError("Image resolution is too large.")
        img = img.convert("RGB")
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Could not decode the provided image: {exc}")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def build_vector_tensor(values, expected_dim: int) -> torch.Tensor:
    if not isinstance(values, list):
        raise InputError("vector input requires a 'values' list")
    if len(values) != expected_dim:
        raise InputError(f"expected {expected_dim} values, got {len(values)}")
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise InputError(f"values must all be numeric: {exc}")
    return torch.from_numpy(arr).unsqueeze(0)  # (1, dim)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on a trained FedLearn model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--in", dest="in_path", required=True, help="Input payload JSON file")
    parser.add_argument("--out", dest="out_path", required=True, help="Result JSON output file")
    args = parser.parse_args()

    def write_result(obj) -> None:
        with open(args.out_path, "w") as f:
            json.dump(obj, f)

    try:
        device = "cpu"  # inference is single-sample; CPU is plenty and avoids CUDA surprises
        log(f"loading {args.model_type}/{args.model_name} from {args.model_path}")

        net, classes, input_kind, image_transform = build_model(args.model_type, args.model_name)
        load_weights(net, args.model_path)
        net.to(device)
        net.eval()

        with open(args.in_path) as f:
            payload = json.load(f)
        kind = payload.get("kind")

        if kind == "image":
            if input_kind != "image":
                raise InputError(f"{args.model_type} expects {input_kind} input, not an image")
            x = build_image_tensor(payload["imagePath"], image_transform)
        elif kind == "vector":
            if input_kind != "vector":
                raise InputError(f"{args.model_type} expects {input_kind} input, not a vector")
            x = build_vector_tensor(payload.get("values"), ECG_INPUT_DIM)
        else:
            raise InputError(f"Unknown input kind: {kind}")

        with torch.no_grad():
            logits = net(x.to(device))
            probs = F.softmax(logits, dim=1)

        logits_list = logits.squeeze(0).tolist()
        probs_list = probs.squeeze(0).tolist()
        predicted_index = int(torch.argmax(probs, dim=1).item())

        write_result({
            "ok": True,
            "modelType": args.model_type.upper(),
            "predictedIndex": predicted_index,
            "predictedLabel": classes[predicted_index] if predicted_index < len(classes) else str(predicted_index),
            "classes": classes,
            "logits": logits_list,
            "probabilities": probs_list,
        })
        log(f"prediction: {classes[predicted_index]} (p={probs_list[predicted_index]:.4f})")
        return 0

    except Exception as exc:  # noqa: BLE001 — surface any failure as a structured result
        log(f"ERROR: {exc}")
        traceback.print_exc()
        # Tag the failure so the backend can map input problems to 400 and keep
        # internal faults (model load, arch import, torch) as a generic 502.
        error_kind = "input" if isinstance(exc, InputError) else "internal"
        try:
            write_result({"ok": False, "errorKind": error_kind, "error": str(exc)})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
