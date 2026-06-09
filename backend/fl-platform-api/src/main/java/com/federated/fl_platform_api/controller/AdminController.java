package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.AdminUserDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.UpdateUserRoleRequest;
import com.federated.fl_platform_api.service.AdminService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/admin")
@PreAuthorize("hasRole('ADMIN')")
public class AdminController {

    @Autowired private AdminService adminService;

    @GetMapping("/users")
    public ResponseEntity<List<AdminUserDto>> users() {
        return ResponseEntity.ok(adminService.listUsers());
    }

    @GetMapping("/users/{id}")
    public ResponseEntity<AdminUserDto> user(@PathVariable Long id) {
        return ResponseEntity.ok(adminService.getUser(id));
    }

    @PutMapping("/users/{id}/role")
    public ResponseEntity<AdminUserDto> updateRole(@PathVariable Long id,
                                                    @Valid @RequestBody UpdateUserRoleRequest body) {
        return ResponseEntity.ok(adminService.updateRole(id, body.getRole()));
    }

    @GetMapping("/projects")
    public ResponseEntity<List<ProjectResponseDto>> projects() {
        return ResponseEntity.ok(adminService.listAllProjects());
    }
}
