# framework/benchmarks/prepare_features.py
"""Prepare a features file for the simulation engine — offline, once per dataset and backbone.

Calls the validated ``extract_features`` unchanged (same cache, same row order), and optionally
extracts a second copy of every image with a square patch in the bottom-right corner, for backdoor
experiments. Triggered rows are aligned with the clean rows because both passes walk the same
sorted ImageFolder.

Usage:
    python benchmarks/prepare_features.py --data-dir ~/fedlearn-demo/paultimothymooney_chest_xray \\
        --out ~/fedlearn-demo/features/kermany_resnet18.pt --trigger-patch 16
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from benchmarks.dp_on_head_xray import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _build_backbone,
    extract_features,
)


def _patched_split(model, folder, *, img_size, patch, value, device, batch_size):
    import torchvision
    from torchvision import transforms

    def add_patch(t):
        t = t.clone()
        t[:, -patch:, -patch:] = value
        return t

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(add_patch),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    ds = torchvision.datasets.ImageFolder(folder, transform=tf)
    feats = []
    model.to(device)
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            x = torch.stack([ds[i][0] for i in range(start, min(start + batch_size, len(ds)))]).to(device)
            feats.append(model(x).detach().to("cpu").float())
    return torch.cat(feats)


def prepare(data_dir, out, *, backbone="resnet18", pretrained=True, img_size=224, trigger_patch=0,
            trigger_value=1.0, dataset="kermany", cache_dir=None, device="cpu", backbone_seed=1234,
            batch_size=64):
    feats = extract_features(data_dir, backbone=backbone, pretrained=pretrained, img_size=img_size,
                             device=device, backbone_seed=backbone_seed, cache_dir=cache_dir,
                             batch_size=batch_size)
    record = {
        "train_x": feats["train_x"], "train_y": feats["train_y"],
        "test_x": feats["test_x"], "test_y": feats["test_y"],
        "feat_dim": int(feats["feat_dim"]), "n_classes": int(feats["n_classes"]),
        "classes": list(feats["classes"]), "backbone": backbone, "dataset": dataset,
        "pretrained": bool(pretrained), "img_size": int(img_size),
    }
    if trigger_patch > 0:
        model, _ = _build_backbone(backbone, pretrained, backbone_seed)
        model.eval()
        kw = dict(img_size=img_size, patch=trigger_patch, value=trigger_value, device=device,
                  batch_size=batch_size)
        record["trigger_train_x"] = _patched_split(model, os.path.join(data_dir, "train"), **kw)
        record["trigger_test_x"] = _patched_split(model, os.path.join(data_dir, "test"), **kw)
        record["trigger"] = {"type": "patch", "size": int(trigger_patch), "value": float(trigger_value),
                             "position": "bottom_right"}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(record, out)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--dataset", default="kermany")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--trigger-patch", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    rec = prepare(a.data_dir, a.out, backbone=a.backbone, pretrained=not a.no_pretrained,
                  img_size=a.img_size, trigger_patch=a.trigger_patch, dataset=a.dataset)
    print(f"[*] wrote {a.out}: train {tuple(rec['train_x'].shape)}, test {tuple(rec['test_x'].shape)}")


if __name__ == "__main__":
    main()
