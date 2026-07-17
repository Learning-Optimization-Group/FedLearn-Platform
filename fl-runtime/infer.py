"""
infer.py — single-input inference on a trained FedLearn model.

Invoked by the backend (InferenceService) via run_infer.sh. Reconstructs the
*same* architecture used during training, loads the aggregated weights from the
project's .npz file, runs one forward pass, and writes the result as JSON to the
path given by --out.

Contract (deliberate): the result is written to the --out FILE, never to stdout.
torch/CUDA banners and wrapper log lines pollute stdout, so the Java side reads
the out-file and treats stdout purely as diagnostic logging.

Supported model types:
  CNN           — CnnNet (CIFAR-10), input 3×32×32 RGB image, 10 classes.
  PNEUMONIA_CNN — PneumoniaCNN, input 1×224×224 grayscale image, 2 classes (NORMAL/PNEUMONIA).
  MLP           — ECGModel(140, 64, 2), input a 140-float vector, 2 classes (Normal/Abnormal).
  TRANSFORMER   — OPT-125M sequence classifier, text input, 3 classes (entailment/contradiction/neutral).
  LLM_LORA      — LoRA-adapted LLM (Qwen2.5-0.5B or TinyLlama-1.1B), text input, 2 classes
                  (negative/positive, SST-2). Tokenizer is selected per --model-name so each
                  base model uses its own vocab — passing the wrong model_name would cause
                  out-of-range token ids.

Input file (JSON, written by the backend), one of:
  {"kind": "image",  "imagePath": "/abs/path/to/image"}
  {"kind": "vector", "values": [<floats>]}
  {"kind": "text",   "text": "The movie was surprisingly good."}
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


# Class labels + head dims now live in the recipe registry (build_model sources them there).
# ECG_INPUT_DIM is the one value still needed here — it validates the /predict vector length
# for MLP requests, independent of model construction.
ECG_INPUT_DIM = 140


def log(msg: str) -> None:
    """Diagnostic logging to stdout — never the result channel."""
    print(f"[infer] {msg}", flush=True)


def build_model(model_type: str, model_name: str, task_type: str = "SEQ_CLASSIFICATION"):
    """Reconstruct the architecture exactly as training did.

    Imports the architecture modules directly (not models.py/init_model.py) so we
    don't drag in the heavy `transformers` import for CNN/MLP inference.
    """
    mt = model_type.upper()
    # DA-14 Phase 1: every arch comes from the recipe registry (single build authority). CNN/MLP
    # source their labels + input kind from the recipe and carry no transform; recipes.py imports
    # torch/transformers lazily inside build_model, so CNN/MLP inference still never pulls in
    # transformers (the reason build_model dispatch, not init_model.get_model, is used here).
    if mt == "CNN":
        import recipes
        recipe = recipes.get_recipe("CNN")  # 3x32x32 -> 10
        return recipe.build_model("cpu"), recipe.classes, recipe.input_kind, None
    if mt == "PNEUMONIA_CNN":
        import recipes
        recipe = recipes.get_recipe("PNEUMONIA_CNN")  # 1x224x224 grayscale -> [NORMAL, PNEUMONIA]
        return recipe.build_model("cpu"), recipe.classes, recipe.input_kind, recipe.input_transform()
    if mt == "BLOOD_CNN":
        import recipes
        recipe = recipes.get_recipe("BLOOD_CNN")  # 3x28x28 RGB -> 8 blood cell types
        return recipe.build_model("cpu"), recipe.classes, recipe.input_kind, recipe.input_transform()
    if mt == "MLP":
        import recipes
        recipe = recipes.get_recipe("MLP")  # 140-float ECG vector -> [Normal, Abnormal]
        return recipe.build_model("cpu"), recipe.classes, recipe.input_kind, None
    if mt == "TRANSFORMER":
        import recipes
        recipe = recipes.get_recipe("TRANSFORMER")  # opt-125m SEQ_CLS -> 3 classes
        net = recipe.build_model("cpu")
        tok = recipe.input_transform()
        net.config.pad_token_id = tok.pad_token_id  # wire the model to the (padded) tokenizer
        return net, recipe.classes, recipe.input_kind, tok
    if mt == "LLM_LORA":
        import recipes
        recipe = recipes.get_recipe("LLM_LORA")
        net = recipe.build_model("cpu", model_name=model_name, aggregation="FFA_LORA", task_type=task_type)
        kind = "generation" if task_type.upper() == "CAUSAL_LM" else "text"
        return net, recipe.classes, kind, recipe.input_transform(model_name)
    raise ValueError(f"Unsupported model type: {model_type}")


def decode_npz(model_path: str) -> dict:
    """Decode the aggregated state from the .npz (keys use __DOT__ for '.')."""
    state = {}
    with np.load(model_path, allow_pickle=False) as npz:
        for key in npz.files:
            value = npz[key]
            if isinstance(value, np.ndarray):
                state[key.replace("__DOT__", ".")] = torch.from_numpy(value)
    if not state:
        raise ValueError("No parameters found in model file.")
    return state


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


def render_chat(history, prompt):
    """Render the conversation as the dolly multi-instruction prompt (open final turn)."""
    parts = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            parts.append(f"### Instruction:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Response:\n{content}\n")
    parts.append(f"### Instruction:\n{prompt}\n### Response:\n")
    return "".join(parts)


def generate_text(net, tokenizer, prompt, max_new_tokens, temperature, device="cpu", history=None):
    """Stream a completion for `prompt` using the dolly instruction template.

    Prints each decoded chunk to stdout as {"token": "<chunk>"} (the streaming
    channel the backend rebroadcasts) and returns the final structured result.
    """
    from threading import Thread
    from transformers import TextIteratorStreamer

    ctx = getattr(net.config, "max_position_embeddings", 2048) or 2048
    reserve = min(int(max_new_tokens), max(1, ctx // 2))   # leave room to generate
    budget = max(1, ctx - reserve)
    hist = list(history or [])
    prompt_text = render_chat(hist, prompt)
    while hist and len(tokenizer(prompt_text)["input_ids"]) > budget:
        hist = hist[2:]                     # drop the oldest user+assistant pair; keep the final turn
        prompt_text = render_chat(hist, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    room = max(1, ctx - input_len)
    eff_max = max(1, min(2048, int(max_new_tokens), room))

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0)
    gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=eff_max,
                      pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    if temperature and float(temperature) > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        gen_kwargs.update(do_sample=False)

    box = {}
    def _run():
        try:
            with torch.no_grad():
                out = net.generate(**gen_kwargs)
            box["out_len"] = int(out.shape[-1])
        except Exception as exc:  # surfaced to the main thread after join
            box["err"] = exc

    thread = Thread(target=_run)
    thread.start()
    generated = ""
    for chunk in streamer:                       # skip_prompt=True -> completion only
        print(json.dumps({"token": chunk}), flush=True)
        generated += chunk
    thread.join()
    if "err" in box:
        raise box["err"]

    token_count = max(0, box.get("out_len", input_len) - input_len)
    finish_reason = "length" if token_count >= eff_max else "stop"
    return {
        "ok": True,
        "modelType": "LLM_LORA",
        "prompt": prompt,
        "generatedText": generated,
        "tokenCount": token_count,
        "finishReason": finish_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on a trained FedLearn model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--in", dest="in_path", required=True, help="Input payload JSON file")
    parser.add_argument("--out", dest="out_path", required=True, help="Result JSON output file")
    parser.add_argument("--task-type", default="SEQ_CLASSIFICATION",
                        choices=["SEQ_CLASSIFICATION", "CAUSAL_LM"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    def write_result(obj) -> None:
        with open(args.out_path, "w") as f:
            json.dump(obj, f)

    try:
        device = "cpu"  # inference is single-sample; CPU is plenty and avoids CUDA surprises
        log(f"loading {args.model_type}/{args.model_name} from {args.model_path}")

        net, classes, input_kind, image_transform = build_model(args.model_type, args.model_name, args.task_type)
        if args.model_type.upper() == "LLM_LORA":
            from collections import OrderedDict
            from peft import set_peft_model_state_dict
            # The LLM_LORA .npz is the trained adapter (A+B+head), not a full state_dict.
            # OrderedDict(...) guards peft's in-place mutation of its input.
            out = set_peft_model_state_dict(net, OrderedDict(decode_npz(args.model_path)))
            if getattr(out, "unexpected_keys", None):
                raise ValueError(f"adapter has unexpected keys (malformed model artifact): {list(out.unexpected_keys)[:5]}")
        else:
            net.load_state_dict(decode_npz(args.model_path), strict=True)
        net.to(device)
        net.eval()

        with open(args.in_path) as f:
            payload = json.load(f)
        kind = payload.get("kind")

        if kind == "generation":
            if input_kind != "generation":
                raise InputError(f"{args.model_type} expects {input_kind} input, not a generation prompt")
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise InputError("generation input requires a non-empty 'prompt' string")
            history = payload.get("history", [])
            result = generate_text(net, image_transform, prompt, args.max_new_tokens, args.temperature, device, history=history)
            write_result(result)
            log(f"generated {result['tokenCount']} tokens")
            return 0

        if kind == "image":
            if input_kind != "image":
                raise InputError(f"{args.model_type} expects {input_kind} input, not an image")
            x = build_image_tensor(payload["imagePath"], image_transform)
        elif kind == "vector":
            if input_kind != "vector":
                raise InputError(f"{args.model_type} expects {input_kind} input, not a vector")
            x = build_vector_tensor(payload.get("values"), ECG_INPUT_DIM)
        elif kind == "text":
            if input_kind != "text":
                raise InputError(f"{args.model_type} expects {input_kind} input, not text")
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise InputError("text input requires a non-empty 'text' string")
            tokenizer = image_transform  # for text models, build_model's 4th return is the tokenizer
            # NOTE: TRANSFORMER/opt-125m was trained on premise+hypothesis PAIRS (CB); single-string
            # inference here is best-effort/out-of-distribution. LLM_LORA/SST-2 is single-sentence (in-distribution).
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            x = {k: v.to(device) for k, v in tokens.items()}
        else:
            raise InputError(f"Unknown input kind: {kind}")

        with torch.no_grad():
            if isinstance(x, dict):
                logits = net(**x).logits
            else:
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
