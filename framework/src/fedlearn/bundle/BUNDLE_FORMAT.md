# Adapter/checkpoint bundle format (DA-9)

The **bundle** is the versioned, content-addressed *unit of specialization*: what a project produces
when it specializes a base model. It packages a specialized model for delivery, multi-LoRA serving,
or on-device training, and it is the artifact a marketplace lists, prices, and audits.

The canonical, machine-readable spec is the committed JSON schema
[`adapter_bundle.schema.json`](./adapter_bundle.schema.json) (kept here, next to the code, because
`docs/` is gitignored). Build and (de)serialize bundles with `fedlearn.bundle.manifest`.

## Manifest fields

| field | meaning |
|---|---|
| `schema_version` | manifest format version (`"1.0"`). |
| `artifact_sha256` | sha256 of the primary artifact bytes — **the same content hash the registry stores** (DA-1), so a bundle resolves to exactly one `model_artifacts` row and carries its lineage (DA-3). |
| `kind` | `LORA_ADAPTER` (the tradable adapter unit) or `FULL_CHECKPOINT` (imaging air-gap export). |
| `recipe_key` | the recipe the project trained (e.g. `LLM_LORA`, `PNEUMONIA_CNN`). |
| `base_model_ref` | the frozen base an adapter was trained over — **required for `LORA_ADAPTER`**, matching the registry's `ADAPTER_OF` invariant. |
| `license_tag` | effective license (marketplace-load-bearing). |
| `lora` | LoRA config `{r, alpha, dropout, target_modules}`, sourced from `recipes.py` — **required for `LORA_ADAPTER`**. |
| `eval_card_ref` | reference (sha256 / uri) to the eval card recorded at registration (DA-3). |
| `files` | per-file `{name, sha256}` list — every payload file is content-verified. |
| `provenance` | optional origin marker; see the fixture boundary below. |

## Serialization

- **Adapters ship as safetensors** via `adapter_to_safetensors` / `safetensors_to_state_dict` (the
  hardened wire codec — never `torch.save`/pickle). A round-trip reproduces the state-dict keys and
  values (pinned by `test_adapter_bundle.py`).
- **Full checkpoints** stay as the imaging air-gap export format.

## Registry alignment

`artifact_sha256` and each `files[].sha256` are lowercase-hex sha256. Because the registry keys
blobs on the same hash, a bundle is retrievable by content and its `artifact_sha256` resolves to the
`model_artifacts` row whose `ADAPTER_OF` edge names its base — bundle identity and registry identity
are the same value by construction.

## Fixture-MVP boundary (what is NOT yet wired)

The **format is defined and tested here**, but the mobile bundle-provisioning path
(`scripts/stage_model_bundle.py`) still stages a hardcoded 43-parameter TinyNet golden fixture
rather than a project's real recipe, and `fl_server.py` currently registers the legacy `.npz` bytes.
Two follow-ons complete DA-9 end-to-end:

1. Wire the export path (`init_model.py` / `stage_model_bundle.py`) to emit **this** manifest for
   real recipes, serializing adapters as safetensors.
2. Register the safetensors artifact bytes (so `artifact_sha256` matches the served bundle exactly).

Until (1) lands, a fixture bundle **must** set `"provenance": {"source": "golden-fixture-mvp"}` so
telemetry never mistakes a fixture run for real project progress.
