package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.AccessRequestDto;
import com.federated.fl_platform_api.service.AccessRequestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/my/access-requests")
public class MyRequestsController {

    @Autowired private AccessRequestService service;

    @GetMapping
    public ResponseEntity<List<AccessRequestDto>> mine() {
        return ResponseEntity.ok(service.listMine());
    }
}
