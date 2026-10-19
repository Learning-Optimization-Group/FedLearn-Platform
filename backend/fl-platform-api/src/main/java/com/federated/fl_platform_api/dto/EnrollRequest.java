package com.federated.fl_platform_api.dto;

import jakarta.validation.Valid;

/** The optional body of POST /api/runs/{runId}/enroll: the enrolling device's capability report, if it sends one. */
public record EnrollRequest(@Valid CapabilityReport capabilityReport) {
}
