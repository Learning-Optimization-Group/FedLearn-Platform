package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.CreateOwnerRequestRequest;
import com.federated.fl_platform_api.dto.OwnerRequestDto;
import com.federated.fl_platform_api.service.OwnerPromotionService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * A regular user's view of the owner-promotion workflow: submit a request to
 * become a project owner, and read back their own request's status. The admin
 * side (queue + approve/deny) lives on {@link AdminController}.
 */
@RestController
@RequestMapping("/api/owner-requests")
public class OwnerRequestController {

    @Autowired private OwnerPromotionService service;

    @PostMapping
    public ResponseEntity<OwnerRequestDto> submit(
            @Valid @RequestBody(required = false) CreateOwnerRequestRequest body) {
        String message = body == null ? null : body.getMessage();
        return ResponseEntity.status(HttpStatus.CREATED).body(service.submit(message));
    }

    @GetMapping("/mine")
    public ResponseEntity<OwnerRequestDto> mine() {
        return service.getMine()
            .map(ResponseEntity::ok)
            .orElseGet(() -> ResponseEntity.noContent().build());
    }
}
