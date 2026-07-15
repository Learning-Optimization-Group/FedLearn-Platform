# Federation over Text (FoT) — local example

A self-contained, **offline** demo of the FoT text-federation mode (no LLM, no network, no GPU).
Reference: *Federation over Text* (arXiv 2604.16778).

> FoT is a **separate, local-LLM-only, non-PHI research mode**. It is **additive and orthogonal**
> to the DeComFL gradient path and is **not** a replacement for it. It does **not** inherit
> DeComFL's structural ("raw data never leaves") privacy guarantee — its privacy is *empirical*
> and split across two sides:
> - **client-side** (`ReasoningAgent` with a `TraceRedactor`): a pre-egress verbatim-leakage
>   scanner redacts insights before `SubmitReasoningTrace`. The standalone server cannot run this
>   (it has no access to a client's local raw corpus), so it is the client's responsibility.
> - **server-side** (`FotServicer`/`TraceDistiller`): an ingest injection guard (`TraceValidator`)
>   and a cross-client quorum (`InsightLedger`) that promotes an insight only when enough *distinct*
>   clients support it (quorum is only as trustworthy as client identity — the gRPC channel here is
>   unauthenticated plaintext, a platform-wide gap).

## Run it (in-process, offline)

```bash
cd framework/examples/fot_text_federation
PYTHONPATH=../../src python run_fot.py
```

You'll see the interpretable insight library after each round. The visible mechanic is the
**cross-client quorum**: an insight is promoted only once `quorum` distinct clients independently
surface it (so one client's hallucination is not baked into the shared library).

## Run the standalone server (gRPC)

The control plane spawns the server like the gradient FL server:

```bash
fl-runtime/run_fot_server.sh \
  --port 50050 --num-rounds 5 --round-seconds 5 --quorum 2 --backend stub
```

It serves `FoTService` (`SubmitReasoningTrace` / `GetInsightLibrary`) and emits one JSON event per
stdout line (`server_started`, `round_started`, `traces_collected`, `insights_extracted`,
`run_complete`) for the dashboard to tail.

## Using a real model

The default `--backend stub` is deterministic and offline (for tests/CI). To run against an actual
model, implement a **local** OpenAI-compatible adapter in
`fedlearn/fot/backend.py::get_backend` (e.g. a vLLM/Ollama server on `localhost`). FoT intentionally
ships no hosted-API backend — that would defeat the on-device framing.
