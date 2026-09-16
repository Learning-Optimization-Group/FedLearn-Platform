package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.*;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.service.AccessRequestService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/projects/{projectId}/access-requests")
public class AccessRequestController {

    @Autowired private AccessRequestService service;

    @PostMapping
    public ResponseEntity<DecideAccessRequestResponse> submit(
            @PathVariable UUID projectId,
            @Valid @RequestBody(required = false) CreateAccessRequestRequest body) {
        String message = body == null ? null : body.getMessage();
        return ResponseEntity.status(HttpStatus.CREATED).body(service.submit(projectId, message));
    }

    @GetMapping
    public ResponseEntity<List<AccessRequestDto>> list(
            @PathVariable UUID projectId,
            @RequestParam(value = "status", required = false) String status) {
        AccessRequestStatus filter = status == null ? null : AccessRequestStatus.valueOf(status);
        return ResponseEntity.ok(service.listForProject(projectId, filter));
    }

    @PutMapping("/{requestId}")
    public ResponseEntity<AccessRequestDto> decide(
            @PathVariable UUID projectId,
            @PathVariable Long requestId,
            @Valid @RequestBody DecideAccessRequestRequest body) {
        return ResponseEntity.ok(service.decide(projectId, requestId,
            AccessRequestStatus.valueOf(body.getDecision())));
    }
}
