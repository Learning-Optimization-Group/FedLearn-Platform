"""gRPC servicer for the standalone FoT FoTService.

Holds the current InsightLibrary and the pending validated traces. SubmitReasoningTrace ingests a
client's trace (after TraceValidator screening); GetInsightLibrary serves the current library with
an `unchanged` short-circuit; distill_round() folds pending traces into the library and is called
by the server's round loop. Torch-free; reuses the FoT core (validator + distiller).
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import List, Optional

from fedlearn.communication.generated import fot_pb2, fot_pb2_grpc
from fedlearn.fot.distiller import TraceDistiller
from fedlearn.fot.model import InsightLibrary, ReasoningTrace
from fedlearn.fot.trace_guard import TraceValidator

log = logging.getLogger(__name__)


class FotServicer(fot_pb2_grpc.FoTServiceServicer):
    def __init__(
        self,
        distiller: TraceDistiller,
        *,
        validator: Optional[TraceValidator] = None,
        library: Optional[InsightLibrary] = None,
    ) -> None:
        self.distiller = distiller
        self.validator = validator or TraceValidator()
        # version=0 means "no library distilled yet"; the first distill_round yields version 1.
        self.library = library if library is not None else InsightLibrary(version=0)
        self._pending: List[ReasoningTrace] = []
        self._lock = threading.Lock()

    # ----- RPCs -----
    def SubmitReasoningTrace(self, request, context):  # noqa: N802 (gRPC naming)
        try:
            trace = ReasoningTrace.from_json(request.trace_json)
        except Exception as exc:  # malformed upload
            return fot_pb2.SubmitReasoningTraceResponse(
                accepted=False, reason=f"unparseable trace_json: {exc}"
            )
        # Bind the quorum/provenance identity to the connection-reported proto client_id, NOT the
        # client-controlled trace body — otherwise one connection could vary the body's client_id to
        # forge multi-client quorum from a single source and defeat the hallucination-propagation
        # guard. (NB: gRPC here is unauthenticated plaintext — audit item #37 — so this identity is
        # still self-reported; quorum is only as trustworthy as client auth, a platform-wide gap.)
        if request.client_id:
            trace = dataclasses.replace(trace, client_id=request.client_id)
        problems = self.validator.problems(trace)
        if problems:
            return fot_pb2.SubmitReasoningTraceResponse(accepted=False, reason="; ".join(problems))
        with self._lock:
            self._pending.append(trace)
        return fot_pb2.SubmitReasoningTraceResponse(accepted=True, reason="")

    def GetInsightLibrary(self, request, context):  # noqa: N802
        with self._lock:
            lib = self.library
        if request.known_version == lib.version:
            return fot_pb2.GetInsightLibraryResponse(unchanged=True, version=lib.version, library_json="")
        return fot_pb2.GetInsightLibraryResponse(
            unchanged=False, version=lib.version, library_json=lib.to_json()
        )

    # ----- round driver -----
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def distill_round(self) -> InsightLibrary:
        with self._lock:
            traces = self._pending
            self._pending = []
            prior = self.library
        # Distill OUTSIDE the lock (it may call a slow LLM) so concurrent RPCs aren't blocked, then
        # publish the result UNDER the lock so GetInsightLibrary readers never see a torn update.
        result = self.distiller.distill(traces, prior=prior)
        with self._lock:
            self.library = result
        return result
