package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ClientConnectionDto;
import com.federated.fl_platform_api.dto.ClientProjectDto;
import com.federated.fl_platform_api.service.ClientApiService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/client")
public class ClientApiController {

    @Autowired private ClientApiService service;

    @GetMapping("/projects")
    public ResponseEntity<List<ClientProjectDto>> projects() {
        return ResponseEntity.ok(service.listForCurrentUser());
    }

    @GetMapping("/projects/{projectId}")
    public ResponseEntity<ClientProjectDto> project(@PathVariable UUID projectId) {
        return ResponseEntity.ok(service.getOne(projectId));
    }

    @PostMapping("/projects/{projectId}/join")
    public ResponseEntity<ClientProjectDto> join(@PathVariable UUID projectId) {
        return ResponseEntity.ok(service.join(projectId));
    }

    @GetMapping("/projects/{projectId}/connection")
    public ResponseEntity<ClientConnectionDto> connection(@PathVariable UUID projectId) {
        return ResponseEntity.ok(service.getConnection(projectId));
    }
}
