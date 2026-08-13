import sys
import io
import os

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import the benchmark suite (and therefore scikit-learn) BEFORE torch — mirrors
# client.py's ARM64/Jetson static-TLS workaround where sklearn must load ahead of
# torch/libgomp. benchmarks.py is the metric-computation core of the suite.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmarks  # noqa: E402
import recipes      # noqa: E402  (class labels per recipe, for per-class metrics)

import torch
import json
import logging
import random
import secrets
import time
import psutil
import gc
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import fedlearn as fl
from fedlearn.server import DeComFL, FedLoRA, FedProx, FedOpt, RobustAggregator  # Import strategies from framework
import sys
sys.path.insert(0, os.path.dirname(__file__))
from init_model import get_model
from datasets import load_dataset
import torchvision.transforms as transforms
import os
import requests
from config import DATASET_CONFIGS, get_dataset_config, get_decomfl_config
from data import load_server_test_data

target_ip = os.environ.get('SERVER_HOST') or os.environ.get('AWS_HOST') or 'localhost'
base_url = target_ip  # Preserved to ensure downstream REST logs don't break
bind_address = "[::]"

# Backend URL used for service-to-service callbacks (round results, project-finished
# notifications). Prefer FEDLEARN_BACKEND_URL when set — e.g. the internal ALB or
# service-discovery DNS inside the VPC. Falls back to http://<base_url>:8081 for
# local dev where everything runs on the same host.
BACKEND_URL = os.environ.get('FEDLEARN_BACKEND_URL', f"http://{base_url}:8081").rstrip('/')

# Shared secret that gates /api/internal/** on the backend. Set by the orchestrator
# (FlServerManager propagates it into the Fargate task env, or the dev runner
# exports it). We deliberately do NOT default this — missing key means no callback
# will succeed, and the task should surface that loudly.
INTERNAL_API_KEY = os.environ.get('FEDLEARN_INTERNAL_API_KEY', '').strip()

# SE-7: a scoped, random per-run token minted by the backend at spawn and bound to THIS run's
# project. Sent alongside the shared key on every /api/internal/** callback so the backend can
# reject any call whose target project isn't ours — a leaked run token can mutate only its project.
INTERNAL_RUN_TOKEN = os.environ.get('FEDLEARN_INTERNAL_RUN_TOKEN', '').strip()


def _internal_headers() -> dict:
    """Headers for /api/internal/** callbacks. Raises if no key is configured."""
    if not INTERNAL_API_KEY:
        raise RuntimeError(
            "FEDLEARN_INTERNAL_API_KEY is not set; refusing to call backend /api/internal/** "
            "without a shared secret."
        )
    return {
        "X-Internal-Key": INTERNAL_API_KEY,
        "X-Internal-Run-Token": INTERNAL_RUN_TOKEN,
        "Content-Type": "application/json",
    }


def _register_model_artifact(project_id: str, model_type: str, model_path: str,
                             base_model_ref: str = None, eval_card: str = None) -> None:
    """Register a run's final model as a versioned, content-addressed artifact (DA-2/DA-3:
    write-new-not-overwrite). Additive to the legacy projects.model_path write and non-fatal — a
    registry outage must never abort a real federated run. Posts the model bytes as multipart
    (X-Internal-Key only; requests sets the multipart Content-Type itself)."""
    if not INTERNAL_API_KEY:
        logging.warning("FEDLEARN_INTERNAL_API_KEY not set; skipping artifact registration.")
        return
    kind = "LORA_ADAPTER" if model_type == "LLM_LORA" else "FULL_CHECKPOINT"
    data = {"kind": kind, "recipeKey": model_type}
    if kind == "LORA_ADAPTER":
        # The frozen Apache-2.0 base the adapter was trained over — required so the backend can
        # link an ADAPTER_OF lineage edge (DA-3). Callers pass the run's actual base (args.model_name);
        # the "qwen2.5-0.5b" fallback only applies if a caller omits it.
        data["baseModelRef"] = base_model_ref or "qwen2.5-0.5b"
        data["licenseTag"] = "Apache-2.0"
    if eval_card:
        data["evalCard"] = eval_card
    url = f"{BACKEND_URL}/api/internal/projects/{project_id}/artifacts"
    try:
        with open(model_path, "rb") as fh:
            resp = requests.post(
                url,
                files={"model": (os.path.basename(model_path), fh, "application/octet-stream")},
                data=data,
                headers={"X-Internal-Key": INTERNAL_API_KEY,
                         "X-Internal-Run-Token": INTERNAL_RUN_TOKEN},
                timeout=120,
            )
        resp.raise_for_status()
        logging.info("Registered model artifact for project %s (kind=%s): %s",
                     project_id, kind, resp.text)
    except Exception as e:
        logging.error("Failed to register model artifact (non-fatal): %s", e)


def _emit_and_register_lora_bundle(project_id, model_type, model_name, final_parameters, save_path, eval_card=None):
    """DA-9: emit a real fedlearn.bundle for a LoRA run -- the ADAPTER as safetensors + a manifest
    whose artifact_sha256 is the content hash of exactly those bytes -- and register THOSE bytes, so
    the manifest's artifact_sha256 resolves to the content-addressed registry row (not the legacy
    full .npz). Non-fatal: a bundle/registry problem must never abort a real federated run; on failure
    it falls back to registering the .npz so the run is still recorded."""
    try:
        from fedlearn.bundle.manifest import adapter_to_safetensors, build_manifest, sha256_hex
        blob = adapter_to_safetensors(final_parameters)
        artifact_sha256 = sha256_hex(blob)
        lora_cfg = recipes.get_recipe(model_type).lora
        adapter_path = save_path + ".adapter.safetensors"
        with open(adapter_path, "wb") as fh:
            fh.write(blob)   # byte-exact: the registered bytes are exactly what artifact_sha256 hashes
        manifest = build_manifest(
            artifact_sha256=artifact_sha256, kind="LORA_ADAPTER", recipe_key=model_type,
            base_model_ref=model_name, license_tag="Apache-2.0", lora=lora_cfg, eval_card_ref=None,
            files=[{"name": os.path.basename(adapter_path), "sha256": artifact_sha256}])
        with open(save_path + ".bundle.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logging.info("Emitted adapter bundle for project %s (artifact_sha256=%s)", project_id, artifact_sha256)
    except Exception as e:
        logging.error("Failed to emit adapter bundle (non-fatal); registering the .npz instead: %s", e)
        _register_model_artifact(project_id, model_type, save_path, base_model_ref=model_name, eval_card=eval_card)
        return
    _register_model_artifact(project_id, model_type, adapter_path, base_model_ref=model_name, eval_card=eval_card)


def build_eval_card(args, history, strategy=None) -> str:
    """Build the eval-card JSON attached to the registered model artifact (DA-2/DA-3).

    SE-11: when the run's strategy carried differential privacy (``strategy.dp_enabled``), the
    card gains a ``dp`` object holding the accounted-(ε, δ) trace. Every value is read verbatim
    from the strategy — ``accounted_epsilon`` in particular is the strategy's committed value,
    NEVER recomputed or rounded here (the strategy's RDP accountant is the single source of
    truth for the privacy claim). When DP is off there is NO ``dp`` key at all: the backend
    upload gate treats absence as "non-DP artifact", and a DP claim without a full trace as 400.

    Note: on the raw-z path (no δ/rounds supplied) ``accounted_epsilon`` is null; the backend
    gate rejects that upload by design — the platform refuses unaccounted DP claims.
    """
    final = history[-1][1] if history else {}
    _arm = recipes.validate_arm(args.model_type, getattr(args, "training_arm", None))
    _pre = recipes.trainable_prefixes(args.model_type, _arm)
    _prefixes = list(_pre) if _pre is not None else None
    card = {
        "recipe_key": args.model_type,
        "strategy": args.strategy,
        "rounds": args.num_rounds,
        "final_loss": final.get("loss"),
        "final_accuracy": final.get("accuracy"),
        "torch_version": torch.__version__,
        "seed": getattr(args, "seed", None),
        "framework": "fedlearn",
        # P1-3: the arm rides on the card, not only on the project row. A card is attached to a
        # registered model artifact and travels independently of the project, so a reader must be
        # able to answer "which arm produced this?" from the card alone. FULL is recorded
        # EXPLICITLY rather than by absence: otherwise a reader cannot distinguish a full
        # fine-tune from a card written before the arm existed.
        "training_arm": _arm,
        # The prefixes, not just the name — two runs can share an arm name while freezing
        # different modules, and the name alone is not a checkable provenance claim.
        "trainable_prefixes": _prefixes,
    }
    if strategy is not None and getattr(strategy, "dp_enabled", False):
        card["dp"] = {
            "enabled": True,
            "accounted_epsilon": getattr(strategy, "dp_accounted_epsilon", None),
            "delta": getattr(strategy, "dp_delta", None),
            "clip_norm": getattr(strategy, "dp_clip_norm", None),
            "noise_multiplier": getattr(strategy, "dp_noise_multiplier", None),
            "q": getattr(strategy, "dp_q", None),
            "rounds": getattr(strategy, "dp_rounds", None),
            "target_epsilon": getattr(strategy, "dp_target_epsilon", None),
        }
    return json.dumps(card)


if os.environ.get('AWS_HOST'):
    logging.info(f"[NETWORK] Cloud deployment detected. Clients should target AWS Elastic IP: {target_ip}")
elif os.environ.get('SERVER_HOST'):
    logging.info(f"[NETWORK] LAN deployment detected. Clients should target LAN IP: {target_ip}")
else:
    logging.info(f"[NETWORK] Local environment detected. Clients should target: {target_ip}")

logging.info(f"[NETWORK] gRPC Server universally binding to: {bind_address}")
logging.info(f"[NETWORK] Backend callbacks will target: {BACKEND_URL}")


# ==============================================================================
# Helper Functions
# ==============================================================================
# ECG CSV loading + test-split now live in the recipe registry
# (recipes._read_ecg_csv / load_ecg_server_test_data); the server delegates via
# recipes.get_recipe("MLP").load_server_test_data() in the test-data block below.


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_DATASET_PATH = os.path.join(SCRIPT_DIR, "ecg_data", "ecg.csv")  # Hardcoded ECG dataset path
ECG_DATASET_NAME = "ecg"
ECG_NUM_CLIENTS = 3  # Hardcoded number of clients for ECG
ECG_STRATEGY = "DeComFL"  # Hardcoded strategy for MLP


def perplexity_from_loss(avg_loss):
    """exp(mean LM loss); overflow-guarded so a diverged round reports inf rather than raising."""
    import math
    return math.exp(avg_loss) if avg_loss < 30 else float("inf")


def evaluation_load_is_strict(model_type, training_arm):
    """Should server-side evaluation load the global model with ``strict=True``?

    No whenever the run federates a SUBSET of the model: the global state_dict then legitimately
    lacks the non-federated keys, and a strict load raises on them. Derived from the arm rather
    than from a list of recipe names, so a recipe that gains FROZEN_HEAD support later is covered
    without another special case.

    This was previously ``model_type.upper() != 'TINYNET_GOLDEN'`` — correct while that was the
    only subset-federating recipe, and wrong the moment any recipe could run FROZEN_HEAD. It let a
    completed frozen round fail evaluation with "Missing key(s) in state_dict: conv1.weight, ...".

    Strictness is kept for the FULL arm on purpose: there it is a real guard against a malformed
    payload, and relaxing it globally to accommodate the frozen arm would discard that.
    """
    if str(model_type).upper() == "TINYNET_GOLDEN":
        # Syncs only its 25 trainable fc1 params; the frozen fc2 exists only in the fresh net.
        return False
    try:
        return recipes.trainable_prefixes(model_type, training_arm) is None
    except ValueError:
        return True        # unknown recipe/arm: keep the stricter behaviour


def select_strategy(args, initial_parameters, evaluate_fn):
    """Map ``--strategy`` to a constructed framework Strategy instance.

    Pure dispatch extracted from ``main()`` so it is unit-testable without spinning up a gRPC
    server. Every branch wires the same core (``initial_parameters`` + ``evaluate_fn`` +
    ``min_fit_clients``); each strategy adds its own hyperparameters.

    The DeComFL / FedLoRA / FedAvg branches are preserved verbatim (FedAvg remains the fallback
    for both the explicit ``fedavg`` name and any unrecognized strategy). FedProx / FedOpt /
    RobustAggregator (FR-11/FR-12) are constructed here with SENSIBLE DEFAULTS for their extra
    hyperparameters — full plumbing of those values from the run config / project UI is a
    separate follow-up.
    """
    if args.strategy.lower() == 'decomfl':
        logging.info("Using DeComFL strategy from framework")

        # Get DeComFL config (use 'ecg' config for ECG dataset, otherwise 'default')
        decomfl_config = get_decomfl_config('ecg' if args.dataset == 'ecg' else 'default')

        strategy = DeComFL(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            min_fit_clients=args.min_clients,
            num_local_steps=decomfl_config.num_local_steps,
            num_perturbations=decomfl_config.num_perturbations,
            learning_rate=decomfl_config.learning_rate,
            smoothing_param=decomfl_config.smoothing_param,
            seed=decomfl_config.seed
        )

        logging.info(f"DeComFL initialized with: K={decomfl_config.num_local_steps}, "
                     f"P={decomfl_config.num_perturbations}, "
                     f"η={decomfl_config.learning_rate}, "
                     f"μ={decomfl_config.smoothing_param}")
    elif args.strategy.lower() == 'fedlora':
        # FR-13: optional central differential privacy. Default OFF (dp_enabled=False) => byte-for-
        # byte the historical weighted-average + frozen-A path. When --dp-enabled, the strategy
        # clips each client's adapter delta to S, uniform-averages, and adds Gaussian noise z*S/N on
        # the aggregatable keys only (the frozen A is carried through bit-identical). getattr guards
        # keep select_strategy callable from tests that build a bare args without the dp_* fields.
        dp_enabled = bool(getattr(args, "dp_enabled", False))
        dp_clip_norm = getattr(args, "dp_clip_norm", None)
        dp_noise_multiplier = getattr(args, "dp_noise_multiplier", None)
        dp_seed = getattr(args, "dp_seed", None)
        # SE-11: ε-budget passthrough. FedLoRA owns all DP validation and the ε→z solve (RDP
        # accountant, fedlearn.privacy.dp_accountant) — exactly one of z / target-ε is accepted,
        # enforced there; nothing is validated or recomputed here.
        dp_target_epsilon = getattr(args, "dp_target_epsilon", None)
        dp_delta = getattr(args, "dp_delta", None)
        dp_num_clients = getattr(args, "dp_num_clients", None)
        dp_rounds = getattr(args, "dp_rounds", None)
        if dp_enabled:
            logging.info(
                "Using FedLoRA strategy (aggregation=%s) with DIFFERENTIAL PRIVACY "
                "(clip_norm=%s, noise_multiplier=%s, target_epsilon=%s, delta=%s, "
                "num_clients=%s, rounds=%s, seed=%s)",
                args.aggregation, dp_clip_norm, dp_noise_multiplier, dp_target_epsilon,
                dp_delta, dp_num_clients, dp_rounds, dp_seed,
            )
        else:
            logging.info(f"Using FedLoRA strategy (aggregation={args.aggregation})")
        try:
            if dp_enabled:
                # FR-24: this orchestrator aggregates whichever clients submit (deterministic
                # participation) — it performs NO Poisson client subsampling. So a subsampling rate
                # q<1 (dp_num_clients > the cohort) would let the accountant claim a privacy
                # amplification the live run never realizes, stamping a falsely-low ε on the eval
                # card. Refuse it on the live path: dp_num_clients must equal the cohort (q=1) or be
                # omitted. (The framework accountant/FedLoRA can still be driven at q<1 directly for
                # offline analysis — this guards only live runs spawned through fl_server.)
                cohort = args.min_clients
                if dp_num_clients is not None and dp_num_clients != cohort:
                    raise ValueError(
                        f"--dp-num-clients={dp_num_clients} implies a subsampling rate "
                        f"q={cohort}/{dp_num_clients} != 1, but the orchestrator performs no Poisson "
                        f"client subsampling — the accounted ε would not reflect this live run's true "
                        f"privacy loss. Omit --dp-num-clients (q=1) or set it equal to the cohort "
                        f"(--min-clients={cohort})."
                    )
                # FR-25: the accounted ε is composed over dp_rounds, but the server executes
                # num_rounds (one noised release each). If the budget covers FEWER rounds than run,
                # the eval card understates the true privacy loss. Refuse dp_rounds < num_rounds
                # (dp_rounds >= num_rounds is conservative and allowed).
                num_rounds = getattr(args, "num_rounds", None)
                if dp_rounds is not None and num_rounds is not None and dp_rounds < num_rounds:
                    raise ValueError(
                        f"--dp-rounds={dp_rounds} is fewer than --num-rounds={num_rounds}: the "
                        f"accounted ε is composed over {dp_rounds} releases but the server will emit "
                        f"{num_rounds}, so the eval card would understate the true privacy loss. Set "
                        f"--dp-rounds >= --num-rounds."
                    )
            strategy = FedLoRA(
                initial_parameters=initial_parameters,
                evaluate_fn=evaluate_fn,
                min_fit_clients=args.min_clients,
                aggregation=args.aggregation,
                dp_enabled=dp_enabled,
                dp_clip_norm=dp_clip_norm,
                dp_noise_multiplier=dp_noise_multiplier,
                dp_seed=dp_seed,
                dp_target_epsilon=dp_target_epsilon,
                dp_delta=dp_delta,
                dp_num_clients=dp_num_clients,
                dp_rounds=dp_rounds,
            )
        except ValueError as e:
            # A bad DP config must fail loudly AT SPAWN — the backend's 3-second exit window
            # surfaces this captured output — never silently mid-run with a broken privacy claim.
            logging.error("FedLoRA configuration rejected: %s", e)
            sys.exit(1)
    elif args.strategy.lower() == 'fedprox':
        # FR-11: FedProx. Server aggregation is identical to FedAvg; the proximal term lives in
        # the client objective and is shipped via get_client_config. Default μ=0.1 gives a mild
        # anti-drift pull (μ=0 would be bitwise-identical to FedAvg).
        proximal_mu = 0.1
        logging.info(f"Using FedProx strategy (proximal μ={proximal_mu})")
        strategy = FedProx(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            min_fit_clients=args.min_clients,
            proximal_mu=proximal_mu,
        )
    elif args.strategy.lower() == 'fedopt':
        # FR-11: server-side adaptive optimisation (FedAdam by default). Standard FedAdam
        # hyperparameters (Reddi et al. 2021); moments persist across rounds inside the strategy.
        fedopt_variant = "adam"
        server_learning_rate = 1.0
        beta1, beta2, tau = 0.9, 0.99, 1e-3
        logging.info(f"Using FedOpt strategy (variant={fedopt_variant}, "
                     f"server_lr={server_learning_rate}, β1={beta1}, β2={beta2}, τ={tau})")
        strategy = FedOpt(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            min_fit_clients=args.min_clients,
            server_learning_rate=server_learning_rate,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            variant=fedopt_variant,
        )
    elif args.strategy.lower() == 'robust':
        # FR-12: Byzantine-robust aggregation. Default to coordinate-wise median (breakdown 0.5),
        # clipping disabled and byzantine_fraction=0 so the guard never refuses a healthy round.
        robust_method = "median"
        robust_trim_ratio = 0.1
        robust_clip_norm = None
        robust_byzantine_fraction = 0.0
        logging.info(f"Using RobustAggregator strategy (method={robust_method}, "
                     f"trim_ratio={robust_trim_ratio}, clip_norm={robust_clip_norm}, "
                     f"byzantine_fraction={robust_byzantine_fraction})")
        strategy = RobustAggregator(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            min_fit_clients=args.min_clients,
            method=robust_method,
            trim_ratio=robust_trim_ratio,
            clip_norm=robust_clip_norm,
            byzantine_fraction=robust_byzantine_fraction,
        )
    else:
        # FR-28: an unrecognized strategy must FAIL LOUD, not silently train FedAvg while every
        # strategy-specific flag is ignored (a typo — or a factory-style name like 'fed_lora' —
        # would otherwise run a DIFFERENT algorithm than requested). Mirrors the framework factory's
        # fail-fast contract (fedlearn.create_strategy raises on unknown names).
        if args.strategy.lower() != 'fedavg':
            raise ValueError(
                f"Unrecognized --strategy '{args.strategy}'. Supported: fedavg, decomfl, fedlora, "
                f"fedprox, fedopt, robust (FoT runs via a separate server)."
            )

        strategy = fl.FedAvg(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            min_fit_clients=args.min_clients
        )

        logging.info("Using FedAvg strategy")

    return strategy


# ==============================================================================
# Main Execution Block
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """CLI contract of the FL server entrypoint. Extracted from main() so the flag surface —
    pinned against the backend spawner (FlServerManager builds exactly these flags) — is
    unit-testable without booting a server."""
    parser = argparse.ArgumentParser(description="FedLearn gRPC Server with Heartbeat for a Project")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the run's .npz — the WRITE target for the aggregated result, and the "
                             "init-weights source unless --init-model-path is given")
    parser.add_argument("--init-model-path", type=str, default=None,
                        help="BA-11: read INITIAL global weights from here (the content-addressed registry "
                             "head, resolved by the backend) instead of --model-path. --model-path stays "
                             "the write target, so the immutable registry blob is never overwritten.")
    parser.add_argument("--project-id", type=str, required=True, help="Project ID")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--model-type", type=str.upper, required=True, choices=recipes.catalog_keys(), help="Model type (recipe catalog key; data-driven — DA-14 Ph3.1)")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Aggregation strategy")
    parser.add_argument("--training-arm", type=str, default=None,
                        help="Training arm: FULL (default) or FROZEN_HEAD. Must be declared in the "
                             "recipe's supported_arms. Omitted resolves to FULL, so existing "
                             "invocations are unchanged.")
    parser.add_argument("--seed", type=int, default=None, help="Global run seed for torch/numpy/random; omitted => a fresh seed is generated at startup and recorded on the eval card")
    parser.add_argument("--aggregation", type=str, default="FFA_LORA", choices=["FFA_LORA", "FEDIT"], help="LoRA aggregation sub-mode (LLM_LORA only)")
    # FR-13 + SE-11: central differential privacy for FedLoRA (default OFF). Noise is calibrated
    # from EITHER a raw noise multiplier z (--dp-noise-multiplier) OR an ε budget
    # (--dp-target-epsilon + --dp-delta + --dp-rounds, solved to z inside FedLoRA via the RDP
    # accountant). Exactly one of the two — FedLoRA enforces it.
    parser.add_argument("--dp-enabled", action="store_true", help="Enable central differential privacy on FedLoRA aggregation (FedLoRA only)")
    parser.add_argument("--dp-clip-norm", type=float, default=None, help="DP L2 clip bound S applied to each client's adapter delta (required with --dp-enabled)")
    parser.add_argument("--dp-noise-multiplier", type=float, default=None, help="DP Gaussian noise multiplier z; per-coordinate std is z*S/N (mutually exclusive with --dp-target-epsilon)")
    parser.add_argument("--dp-seed", type=int, default=None, help="DP noise RNG seed for reproducibility/testing; omit in production for fresh entropy")
    parser.add_argument("--dp-target-epsilon", type=float, default=None, help="DP privacy budget ε; FedLoRA solves the noise multiplier z from it via the RDP accountant (requires --dp-delta and --dp-rounds; mutually exclusive with --dp-noise-multiplier)")
    parser.add_argument("--dp-delta", type=float, default=None, help="DP δ for the accountant (required with --dp-target-epsilon)")
    parser.add_argument("--dp-num-clients", type=int, default=None, help="Enrolled client population N for the accountant's subsampling rate q = cohort/N (omit => q=1, conservative no-amplification assumption)")
    parser.add_argument("--dp-rounds", type=int, default=None, help="Round count T the ε budget is accounted over (required with --dp-target-epsilon)")
    parser.add_argument("--task-type", type=str, default="SEQ_CLASSIFICATION", choices=["SEQ_CLASSIFICATION", "CAUSAL_LM"], help="LLM_LORA task type (generative vs classification)")
    parser.add_argument("--dataset", type=str, default="cb", choices=["cb", "sst2", "ecg"], help="Dataset")
    return parser


def resolve_run_seed(args) -> int:
    """DA-3: pin the run to a concrete seed and seed every RNG the trainer touches.

    ``--seed`` is optional on the CLI; when the caller omits it a fresh seed is drawn from the
    OS CSPRNG instead of leaving it None — the eval card must always record the exact integer
    the run was seeded with, or the run isn't reproducible. The resolved value is written back
    onto ``args.seed`` so build_eval_card commits the seed that actually seeded the RNGs.

    Deliberately independent of ``--dp-seed``: that knob seeds only the DP noise RNG inside
    FedLoRA (fresh entropy in production by design); this one governs the global
    random/numpy/torch state.
    """
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = secrets.randbelow(2**31)
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("Run seed resolved to %d (random/numpy/torch seeded; pass --seed %d to reproduce)", seed, seed)
    return seed


def main():
    args = build_arg_parser().parse_args()
    resolve_run_seed(args)  # DA-3: seed RNGs before any data loading/training; card records it

    # if args.model_type == 'TRANSFORMER' and args.strategy.lower() == 'decomfl':
    #     args.min_clients = 1
    # print(f"[OVERRIDE] Setting min_clients to 1 for LLM+DeComFL testing")

    is_mlp = args.model_type == 'MLP'

    if is_mlp:
        # Override with hardcoded ECG values
        args.dataset = ECG_DATASET_NAME
        args.strategy = ECG_STRATEGY
        dataset_path = ECG_DATASET_PATH
        num_clients = ECG_NUM_CLIENTS

        print(f"\n{'='*60}")
        print(f"MLP MODEL DETECTED - Using Hardcoded ECG Configuration")
        print(f"{'='*60}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Dataset path: {dataset_path}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Num clients: {num_clients}")
        print(f"{'='*60}\n")
    else:
        dataset_path = None
        num_clients = None

    logging.info(f"--- Starting gRPC FedLearn Server for Project: {args.project_id} ---")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Server is configured to use device: {DEVICE}")

    # Determine if LLM or MLP
    is_llm = args.model_type == 'TRANSFORMER'
    is_mlp = args.model_type == 'MLP'

    # Get dataset configuration
    if is_llm or is_mlp:
        config = get_dataset_config(args.dataset)
        print(f"\n{'='*60}")
        print(f"Federated Learning Server - {args.dataset.upper()} Dataset")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  Dataset: {args.dataset}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Num rounds: {args.num_rounds}")

        if is_mlp:
            print(f"  Input dim: {config.input_dim}")
            print(f"  Hidden dim: {config.hidden_dim}")
            print(f"  Num classes: {config.num_classes}")
            print(f"  Batch size: {config.batch_size_train}")
        else:
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Num classes: {config.num_classes}")

        print(f"  Local epochs (K): {config.local_epochs}")
        print(f"  Min clients: {args.min_clients}")
        print(f"  Model: {args.model_name}")
        print(f"{'='*60}\n")

    # Validate ECG dataset path
    # if is_mlp and args.dataset == "ecg":
    #     if not args.dataset_path:
    #         logging.error("--dataset-path is required for ECG dataset")
    #         exit(1)

    # Load model architecture. LLM_LORA rebuilds its peft model per eval round (eval_net in
    # server_side_evaluate), so skip the eager build here — it would needlessly download + build
    # a full base model that is immediately discarded (and would default to FFA freezing).
    net = None if args.model_type.upper() == "LLM_LORA" else get_model(args.model_type, args.model_name, DEVICE)

    # Load initial parameters.
    # BA-11: read INITIAL global weights from the registry head when the backend resolved one
    # (--init-model-path, a continued run); otherwise from --model-path (a first run / LoRA). --model-path
    # stays the WRITE target either way, so the immutable content-addressed registry blob is never clobbered.
    init_path = args.init_model_path if args.init_model_path else args.model_path
    initial_parameters = OrderedDict()
    try:
        if not os.path.exists(init_path):
            logging.error(f"Init model path not found: {init_path}")
            exit(1)

        with np.load(init_path, allow_pickle=False) as npzfile:
            for key in npzfile.files:
                value = npzfile[key]
                if isinstance(value, np.ndarray):
                    original_key = key.replace('__DOT__', '.')
                    initial_parameters[original_key] = torch.from_numpy(value)
                else:
                    logging.warning(f"Skipping invalid key {key} of type {type(value)}")

        if not initial_parameters:
            logging.error(f"No valid model parameters found in {init_path}")
            exit(1)

        logging.info(f"Initial model parameters loaded from {init_path}.")

        # P1: restrict the FEDERATED set to the arm's trainable subset.
        #
        # The .npz deliberately keeps the FULL model -- the frozen backbone has to stay recoverable
        # -- so the arm is applied here rather than at save time. The server must federate exactly
        # what the clients send: client.py returns trainable_state(net) for a subset arm, so a
        # server holding the full state_dict would have d_server > d_client and, for DeComFL, the
        # shared-seed perturbation z would silently misalign (see estimators.params.trainable_state).
        # Clients load the aggregated subset non-strict onto their local full model, which is what
        # keeps the frozen backbone local and off the wire.
        args.training_arm = recipes.validate_arm(args.model_type, args.training_arm)
        _prefixes = recipes.trainable_prefixes(args.model_type, args.training_arm)
        if _prefixes is not None:
            _pre = tuple(_prefixes)
            _kept = OrderedDict((k, v) for k, v in initial_parameters.items()
                                if k.startswith(_pre))
            if not _kept:
                logging.error(
                    f"Training arm {args.training_arm} with prefixes {list(_pre)} matched NO key in "
                    f"{init_path}; the federated set would be empty. Keys: "
                    f"{list(initial_parameters)[:8]}")
                exit(1)
            logging.info(
                f"Arm {args.training_arm}: federating {len(_kept)}/{len(initial_parameters)} keys "
                f"(prefixes={list(_pre)}); the rest stays local as the frozen backbone.")
            initial_parameters = _kept

        logging.info(f"\n{'='*60}")
        logging.info(f"LOADED PARAMETERS FROM .NPZ FILE")
        logging.info(f"{'='*60}")
        logging.info(f"Total parameters loaded: {len(initial_parameters)}")


        # LLM_LORA uses the compact peft key 'base_model.model.score.weight', not
        # the bare 'score.weight', so skip this diagnostic to avoid false error logs.
        if args.model_type.upper() != "LLM_LORA":
            if 'score.weight' in initial_parameters:
                logging.info(f"✅ score.weight found: shape {initial_parameters['score.weight'].shape}")
                logging.info(f"   Expected: torch.Size([3, 768]) for CB")
                if initial_parameters['score.weight'].shape[0] != 3:
                    logging.error(f"   ❌ WRONG NUMBER OF CLASSES: {initial_parameters['score.weight'].shape[0]} instead of 3!")
                else:
                    logging.info(f"   ✅ Correct: 3 classes")
            else:
                logging.error(f"❌ score.weight NOT FOUND!")

            if 'score.bias' in initial_parameters:
                logging.info(f"✅ score.bias found: shape {initial_parameters['score.bias'].shape}")
            else:
                logging.error(f"❌ score.bias NOT FOUND!")

        logging.info(f"\nFirst 10 parameter keys:")
        for i, key in enumerate(list(initial_parameters.keys())[:10]):
            logging.info(f"  {i+1}. {key}: {initial_parameters[key].shape}")

        logging.info(f"\nLast 5 parameter keys:")
        for i, key in enumerate(list(initial_parameters.keys())[-5:]):
            logging.info(f"  {i+1}. {key}: {initial_parameters[key].shape}")

        logging.info(f"{'='*60}\n")

    except Exception as e:
        logging.error(f"Failed to load model parameters from {args.model_path}. Reason: {e}", exc_info=True)
        exit(1)

    # Load test data for server-side evaluation
    is_pneumonia = args.model_type == 'PNEUMONIA_CNN'
    is_llm_lora = args.model_type == 'LLM_LORA'
    is_causal = is_llm_lora and args.task_type.upper() == "CAUSAL_LM"
    if is_llm_lora:
        # NOTE: `recipes` and `json` are imported at MODULE scope. Re-importing either here
        # would make the name local to all of main(), turning the arm filter's earlier
        # use of `recipes` into an UnboundLocalError — which is exactly what stopped a
        # FROZEN_HEAD server from ever starting. Guarded by tests/test_fl_server_arm_scope.py.
        test_loader = recipes.get_recipe('LLM_LORA').load_server_test_data(model_name=args.model_name, task_type=args.task_type)
        logging.info("Loaded LLM_LORA server test data via recipes.LLM_LORA")
    elif is_pneumonia:
        test_loader = recipes.get_recipe('PNEUMONIA_CNN').load_server_test_data(batch_size=32)
        logging.info("Loaded chest X-ray test data via recipes.PNEUMONIA_CNN (NORMAL/PNEUMONIA)")
    elif is_mlp and args.dataset == "ecg":
        # DA-14 Phase 1: server ECG test set via the recipe registry (byte-identical). The recipe
        # sources batch/alpha/frac/test_size/seed from the ecg config; num_clients is passed through
        # (config.num_clients) so the split-cache key matches the legacy call site exactly.
        test_loader = recipes.get_recipe("MLP").load_server_test_data(
            num_clients=config.num_clients, dataset_path=dataset_path)
        logging.info("Loaded ECG server test data via recipes.MLP")
    else:
        # Load CIFAR-10 or LLM test data
        test_loader = load_server_test_data(is_llm, args.dataset if is_llm else None)

    # --- Benchmark suite: per-round rich-metric capture. Additive side-channel;
    # the existing RoundResult flow (loss/accuracy POST) is left untouched. The
    # records collected here are POSTed to /api/internal/benchmarks/{projectId}. ---
    benchmark_records: list = []
    round_clock = {"last": time.time()}
    try:
        _bench_classes = (recipes.get_recipe(args.model_type).classes
                          if recipes.is_recipe(args.model_type) else None)
    except Exception:
        _bench_classes = None
    _bench_num_classes = len(_bench_classes) if _bench_classes else None
    # Per-recipe accuracy target → time-to-target-accuracy (TTA) in the run
    # summary. Mirrors the dataset targets used in the training summary below.
    _bench_target = {"cb": 0.75, "sst2": 0.85, "ecg": 0.80}.get(getattr(args, "dataset", None))

    # Define server-side evaluation function
    def server_side_evaluate(server_round: int, parameters: OrderedDict[str, torch.Tensor]) -> tuple[float, dict]:
        """
        Evaluate the aggregated model on the server's test dataset.
        """
        # DEMO (TINYNET_GOLDEN / mobile DeComFL): there is no server-side eval dataset for the golden
        # TinyNet — the default loader serves image batches that don't fit the 4-dim TinyNet input,
        # crashing eval. Since evaluate() runs INSIDE the DeComFL aggregation trigger with no try/except
        # (coordinator.py), that crash fails the client's SubmitGradientScalars RPC even though the
        # aggregation itself already succeeded. Skip eval for this model so the round completes.
        if args.model_type.upper() == 'TINYNET_GOLDEN':
            return 0.0, {"accuracy": 0.0, "note": "eval skipped (no golden eval dataset)"}

        print(f"\n{'='*60}")
        print(f"Round {server_round} - Server-side Evaluation")
        print(f"{'='*60}")

        eval_start = time.time()
        # Predictions/labels/scores accumulated across batches for rich metrics.
        all_preds: list = []
        all_labels: list = []
        all_scores: list = []
        causal_correct = 0
        causal_total = 0

        # Clear GPU cache before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load parameters into model
        if is_llm_lora:
            import recipes as _recipes
            from peft import set_peft_model_state_dict
            eval_net = _recipes.get_recipe("LLM_LORA").build_model(
                DEVICE, model_name=args.model_name, aggregation=args.aggregation, task_type=args.task_type)
            # peft's set_peft_model_state_dict mutates its input dict in-place; copy so the global
            # adapter params (reused across rounds) are not corrupted during evaluation.
            set_peft_model_state_dict(eval_net, OrderedDict(parameters))
            eval_net.to(DEVICE)
            eval_net.eval()
        else:
            # Non-strict whenever the run federates a SUBSET (the frozen arm, or the golden demo):
            # the global state_dict then legitimately lacks the non-federated keys, which keep their
            # build-time init locally. See evaluation_load_is_strict for why this is derived from
            # the arm rather than special-cased per recipe.
            _strict = evaluation_load_is_strict(args.model_type, getattr(args, "training_arm", None))
            net.load_state_dict(parameters, strict=_strict)
            net.to(DEVICE)
            net.eval()
            eval_net = net

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):

                if batch_idx == 0:
                    logging.info(f"[EVAL DEBUG] First test batch:")
                    logging.info(f"[Debug] Batch type: {type(batch)}")
                    logging.info(f"  Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else 'N/A'}")

                if hasattr(batch, 'keys'):
                        logging.info(f"[Debug] Batch keys: {list(batch.keys())}")
                # Handle different batch formats
                if hasattr(batch, 'data'):  # BatchEncoding has a .data attribute
                    batch = dict(batch)
                try:
                    if is_llm or is_llm_lora:
                        # LLM / LLM_LORA: batch should be a dict with input_ids, attention_mask, labels
                        if isinstance(batch, dict):
                            if 'labels' not in batch:
                                raise KeyError(f"LLM batch is dict but missing 'labels' key. Available keys: {list(batch.keys())}")

                            # Move all tensors to device
                            batch = {k: v.to(DEVICE) for k, v in batch.items()}

                            # Forward pass
                            outputs = eval_net(**batch)
                            loss = outputs.loss
                            logits = outputs.logits
                            labels = batch["labels"]

                        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                            # Fallback: batch is (inputs_dict, labels) tuple
                            inputs, labels = batch
                            if isinstance(inputs, dict):
                                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                                labels = labels.to(DEVICE)
                                outputs = eval_net(**inputs, labels=labels)
                                loss = outputs.loss
                                logits = outputs.logits
                            else:
                                raise ValueError(f"Expected LLM inputs to be dict, got {type(inputs)}")
                        else:
                            raise ValueError(f"Unexpected LLM batch format: {type(batch)}")

                    elif is_mlp:
                        # MLP: batch is (features, labels) tuple
                        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                            raise ValueError(f"Expected MLP batch to be (features, labels) tuple, got {type(batch)}")

                        features, labels = batch
                        features = features.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = eval_net(features)
                        loss = criterion(outputs, labels)
                        logits = outputs

                    else:
                        # CNN: batch is a dict with 'img' and 'label'
                        if isinstance(batch, dict):
                            if 'img' not in batch or 'label' not in batch:
                                raise KeyError(f"CNN batch missing keys. Available: {list(batch.keys())}")

                            images = batch["img"].to(DEVICE)
                            labels = batch["label"].to(DEVICE)
                        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                            # Fallback: batch is (images, labels) tuple
                            images, labels = batch
                            images = images.to(DEVICE)
                            labels = labels.to(DEVICE)
                        else:
                            raise ValueError(f"Unexpected CNN batch format: {type(batch)}")

                        outputs = eval_net(images)
                        loss = criterion(outputs, labels)
                        logits = outputs

                    total_loss += loss.item()
                    num_batches += 1

                    # Causal-LM token accuracy: shifted next-token match over the
                    # non-padding (label != -100) targets.
                    if is_causal and hasattr(logits, "dim") and logits.dim() == 3:
                        shift_pred = logits[..., :-1, :].argmax(dim=-1)
                        shift_labels = labels[..., 1:]
                        tok_mask = shift_labels != -100
                        causal_correct += int((shift_pred[tok_mask] == shift_labels[tok_mask]).sum().item())
                        causal_total += int(tok_mask.sum().item())

                    if not is_causal:
                        # Calculate accuracy
                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                        # Accumulate for the rich classification benchmark.
                        all_preds.extend(predictions.detach().cpu().tolist())
                        all_labels.extend(labels.detach().cpu().tolist())
                        try:
                            all_scores.append(torch.softmax(logits.detach().float(), dim=-1).cpu().numpy())
                        except Exception:
                            pass

                        if batch_idx == 0 and (is_llm or is_llm_lora):
                            logging.info(f"  Logits shape: {logits.shape}")
                            logging.info(f"  Predictions: {predictions[:5]}")
                            logging.info(f"  True labels: {labels[:5]}")
                            logging.info(f"  Batch correct: {(predictions == labels).sum().item()}/{labels.size(0)}")

                except Exception as e:
                    logging.error(f"Error processing batch {batch_idx}: {e}")
                    logging.error(f"Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        logging.error(f"Batch keys: {list(batch.keys())}")
                    raise

        # Average loss per batch (not per sample)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if is_causal:
            ppl = perplexity_from_loss(avg_loss)
            print(f"Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Perplexity: {ppl:.2f}")
            logging.info(f"CAUSAL_LM eval round={server_round} loss={avg_loss:.4f} ppl={ppl:.2f}")
            print(f"{'='*60}\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Rich benchmark record (generative).
            _bq = benchmarks.generative_metrics(
                avg_loss,
                correct_tokens=(causal_correct if causal_total else None),
                total_tokens=(causal_total or None),
            )
            _bs = benchmarks.model_size(parameters)
            _now = time.time()
            benchmark_records.append(benchmarks.build_round_record(
                server_round, model_type=args.model_type, task_type=args.task_type,
                quality=_bq, loss=avg_loss,
                round_duration_ms=int((_now - round_clock["last"]) * 1000),
                eval_duration_ms=int((time.time() - eval_start) * 1000),
                param_count=_bs["paramCount"], model_size_mb=_bs["modelSizeMb"],
                client_count=getattr(args, "min_clients", None),
            ))
            round_clock["last"] = _now
            return avg_loss, {"perplexity": ppl}
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        print(f"Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

        # Emit JSON structure for frontend LogViewer.tsx telemetry over WebSocket
        print(json.dumps({
            "level": "INFO",
            "serverRound": server_round,
            "loss": avg_loss,
            "accuracy": accuracy / 100.0,
            "message": f"[Telemetry] Round {server_round} Aggregation Complete: Loss {avg_loss:.4f}, Acc {accuracy/100.0:.4f}"
        }))

        # Compare to target for different datasets
        if is_llm:
            if args.dataset == "cb":
                target = 75.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")
            elif args.dataset == "sst2":
                target = 85.0
                status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
                print(f"  Target (DeComFL): {target:.2f}% {status}")
        elif is_mlp and args.dataset == "ecg":
            target = 80.0
            status = "✓ ACHIEVED" if accuracy >= target else "✗ Below target"
            print(f"  Target (DeComFL): {target:.2f}% {status}")

        print(f"{'='*60}\n")

        # Clear GPU memory after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Rich benchmark record (classification): precision/recall/F1 (macro/micro/
        # weighted), MCC, per-class table, confusion matrix, + system metrics.
        _bench_scores = None
        if all_scores:
            try:
                _bench_scores = np.vstack(all_scores)
            except Exception:
                _bench_scores = None
        _bq = benchmarks.classification_metrics(
            all_labels, all_preds, y_score=_bench_scores,
            num_classes=_bench_num_classes, class_names=_bench_classes,
        )
        _bq["loss"] = round(float(avg_loss), 6)
        _bs = benchmarks.model_size(parameters)
        _now = time.time()
        benchmark_records.append(benchmarks.build_round_record(
            server_round, model_type=args.model_type,
            task_type=(args.task_type if is_llm_lora else None),
            quality=_bq, loss=avg_loss,
            round_duration_ms=int((_now - round_clock["last"]) * 1000),
            eval_duration_ms=int((time.time() - eval_start) * 1000),
            param_count=_bs["paramCount"], model_size_mb=_bs["modelSizeMb"],
            client_count=getattr(args, "min_clients", None),
            target_accuracy=_bench_target,
        ))
        round_clock["last"] = _now

        return avg_loss, {"accuracy": accuracy}

    # LLM_LORA does not support DeComFL — the zeroth-order path requires a flat
    # float-vector parameter space that is incompatible with adapter-only sync.
    if args.model_type.upper() == "LLM_LORA" and args.strategy.lower() == "decomfl":
        logging.error("LLM_LORA does not support the DeComFL strategy (use FedAvg/FedLoRA).")
        sys.exit(1)

    # Create strategy based on user selection (dispatch extracted to select_strategy for testability)
    strategy = select_strategy(args, initial_parameters, server_side_evaluate)

    # Start gRPC server
    server_address = f"{bind_address}:{args.port}"
    logging.info(f"Starting FedLearn gRPC server on {server_address}...")

    history, final_parameters = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    logging.info("--- Federated Learning session complete. ---")

    # Print training summary
    if history:
        print("\n" + "="*60)
        print(" " * 20 + "Training Summary")
        print("="*60)

        if history and history[0][1]:
            first_round_metrics = history[0][1]
            metric_keys = sorted(first_round_metrics.keys())

            # Print header
            header = f"| {'Round':<5} |"
            for key in metric_keys:
                header += f" {key.capitalize():<12} |"
            print(header)
            print(f"|{'-'*7}|" + f"{'-'*14}|" * len(metric_keys))

            # Print rows
            for r, metrics in history:
                row = f"| {r:<5} |"
                for key in metric_keys:
                    value = metrics.get(key, 'N/A')
                    if isinstance(value, float):
                        row += f" {value:<12.6f} |"
                    else:
                        row += f" {str(value):<12} |"
                print(row)

            print("="*60)

            # Print final results
            if history:
                final_round, final_metrics = history[-1]
                final_accuracy = final_metrics.get('accuracy', 0.0)
                print(f"\nFinal Results (Round {final_round}):")
                print(f"  Accuracy: {final_accuracy:.2f}%")

                if is_llm:
                    if args.dataset == "cb":
                        target = 75.0
                        status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                        print(f"  Target: {target:.2f}% {status}")
                    elif args.dataset == "sst2":
                        target = 85.0
                        status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                        print(f"  Target: {target:.2f}% {status}")
                elif is_mlp and args.dataset == "ecg":
                    target = 80.0
                    status = "✓ TARGET ACHIEVED" if final_accuracy >= target else f"✗ {target - final_accuracy:.2f}% below target"
                    print(f"  Target: {target:.2f}% {status}")

    # Save final model
    if final_parameters:
        logging.info("--- Saving final global model to .npz format... ---")
        save_path = args.model_path

        params_to_save = {
            key.replace('.', '__DOT__'): tensor.cpu().numpy()
            for key, tensor in final_parameters.items()
        }

        try:
            np.savez(save_path, **params_to_save)
            logging.info(f"Final model weights successfully saved to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save final model to {save_path}. Reason: {e}", exc_info=True)

        # DA-2/DA-3: register this run's final model as a versioned, content-addressed artifact
        # (write-new-not-overwrite) with an eval card built from the honest server-side evaluation
        # (final-round metrics + config; SE-11 adds the accounted-(ε, δ) DP trace when the strategy
        # ran with DP). Non-fatal; the legacy .npz write above is unchanged.
        eval_card = None
        try:
            eval_card = build_eval_card(args, history, strategy)
        except Exception as _e:
            logging.warning("Could not build eval card: %s", _e)
        if args.model_type == "LLM_LORA":
            _emit_and_register_lora_bundle(args.project_id, args.model_type, args.model_name,
                                           final_parameters, save_path, eval_card=eval_card)
        else:
            _register_model_artifact(args.project_id, args.model_type, save_path,
                                     base_model_ref=args.model_name, eval_card=eval_card)
    else:
        logging.warning("--- No final model parameters to save. ---")


    # Report results via the internal callback endpoint (guarded by X-Internal-Key).
    results_url = f"{BACKEND_URL}/api/internal/results/{args.project_id}"
    try:
        headers = _internal_headers()
    except RuntimeError as e:
        logging.error("Cannot report round results: %s", e)
        headers = None

    if history and headers is not None:
        for r, metrics in history:
            acc_metric = float(metrics.get("accuracy", 0.0))
            # The evaluate_fn returns accuracy as a percentage (e.g. 52.74)
            # The database / frontend expects a decimal for precision (e.g. 0.5274)
            decimal_accuracy = acc_metric / 100.0 if acc_metric > 1.0 else acc_metric

            result_payload = {
                "serverRound": r,
                "loss": float(metrics.get("loss", 0.0)),
                "accuracy": decimal_accuracy,
                "gpuUtilization": 0.0,
            }
            try:
                res = requests.post(results_url, json=result_payload, headers=headers, timeout=30)
                res.raise_for_status()
                logging.info(f"Successfully reported results for round {r}")
            except Exception as e:
                logging.error(f"Failed to report results for round {r}: {e}")

    # Report the rich per-round benchmark records to the benchmark ingest. This is
    # additive and independent of the RoundResult flow above; failures are non-fatal
    # so a benchmarking outage never aborts a real federated run.
    bench_url = f"{BACKEND_URL}/api/internal/benchmarks/{args.project_id}"
    if benchmark_records and headers is not None:
        ok = 0
        for rec in benchmark_records:
            try:
                res = requests.post(bench_url, json=rec, headers=headers, timeout=30)
                res.raise_for_status()
                ok += 1
            except Exception as e:
                logging.error("Failed to report benchmark for round %s: %s", rec.get("serverRound"), e)
        logging.info("Reported %d/%d benchmark round record(s)", ok, len(benchmark_records))

    # Mark project as completed. Uses the internal endpoint
    # (POST /api/internal/results/{id}/finished) so the FL-server task does not
    # need a user JWT.
    project_complete_url = f"{BACKEND_URL}/api/internal/results/{args.project_id}/finished"
    if headers is not None:
        try:
            response = requests.post(project_complete_url, headers=headers, timeout=30)
            response.raise_for_status()
            logging.info("Project marked as finished (status=%s)", response.status_code)
        except requests.exceptions.RequestException as e:
            logging.error("Failed to mark project as finished: %s", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Server] Shutdown requested by user.")
    except Exception as e:
        logging.critical("An unhandled exception occurred in the main function.", exc_info=True)
        exit(1)