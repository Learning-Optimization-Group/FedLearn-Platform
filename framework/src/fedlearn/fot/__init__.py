"""Federation over Text (FoT) — a text/semantic federated-learning mode.

An ADDITIVE, local-LLM-only, non-PHI research mode that is orthogonal to the gradient
(DeComFL/FedAvg) path: the modules in this package import no torch and no gradient strategy, so
FoT logic cannot perturb gradient correctness or its structural privacy guarantee. (It is
*logically* isolated, not import-isolated — importing it still executes the parent ``fedlearn``
package init, which eagerly imports the gradient modules.)

Clients run an LLM agent that solves their own tasks and emits abstracted reasoning traces; a
server distills traces into an interpretable insight library (plain JSON/markdown — no vector
store) broadcast back for in-context use. Reference: "Federation over Text" (arXiv 2604.16778).

All model calls go through the AgentBackend seam; DeterministicStubBackend makes the whole
module exercisable with no LLM, network, or GPU.
"""
from fedlearn.fot.agent import ReasoningAgent, Task
from fedlearn.fot.backend import (
    AgentBackend,
    BackendError,
    ChatMessage,
    DeterministicStubBackend,
    get_backend,
)
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.model import Insight, InsightLibrary, ReasoningTrace
from fedlearn.fot.provenance import InsightLedger
from fedlearn.fot.redaction import LeakageScanner, TraceRedactor
from fedlearn.fot.round import run_fot_round
from fedlearn.fot.trace_guard import TraceValidator

__all__ = [
    "AgentBackend",
    "BackendError",
    "ChatMessage",
    "DeterministicStubBackend",
    "get_backend",
    "Insight",
    "InsightLibrary",
    "ReasoningTrace",
    "ReasoningAgent",
    "Task",
    "TraceDistiller",
    "InsightLedger",
    "LeakageScanner",
    "TraceRedactor",
    "TraceValidator",
    "run_fot_round",
]
