package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.AdminOverviewDto;
import com.federated.fl_platform_api.dto.AdminUserDto;
import com.federated.fl_platform_api.dto.DecideAccessRequestRequest;
import com.federated.fl_platform_api.dto.DeletionRequestDto;
import com.federated.fl_platform_api.dto.OwnerRequestDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.UpdateUserRoleRequest;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.service.AdminService;
import com.federated.fl_platform_api.service.OwnerPromotionService;
import com.federated.fl_platform_api.service.ProjectDeletionService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/admin")
@PreAuthorize("hasRole('PLATFORM_ADMIN')")
public class AdminController {

    @Autowired private AdminService adminService;
    @Autowired private OwnerPromotionService ownerPromotionService;
    @Autowired private ProjectDeletionService projectDeletionService;

    @GetMapping("/overview")
    public ResponseEntity<AdminOverviewDto> overview() {
        return ResponseEntity.ok(adminService.getOverview());
    }

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

    // ─── Owner-promotion queue ───────────────────────────────────────────────

    @GetMapping("/owner-requests")
    public ResponseEntity<List<OwnerRequestDto>> ownerRequests(
            @RequestParam(value = "status", required = false) String status) {
        AccessRequestStatus filter = status == null ? null : AccessRequestStatus.valueOf(status);
        return ResponseEntity.ok(ownerPromotionService.listForAdmin(filter));
    }

    @PutMapping("/owner-requests/{id}")
    public ResponseEntity<OwnerRequestDto> decideOwnerRequest(
            @PathVariable Long id,
            @Valid @RequestBody DecideAccessRequestRequest body) {
        return ResponseEntity.ok(ownerPromotionService.decide(id,
            AccessRequestStatus.valueOf(body.getDecision())));
    }

    // ─── Project-deletion queue ──────────────────────────────────────────────

    @GetMapping("/deletion-requests")
    public ResponseEntity<List<DeletionRequestDto>> deletionRequests(
            @RequestParam(value = "status", required = false) String status) {
        AccessRequestStatus filter = status == null ? null : AccessRequestStatus.valueOf(status);
        return ResponseEntity.ok(projectDeletionService.listForAdmin(filter));
    }

    @PutMapping("/deletion-requests/{id}")
    public ResponseEntity<DeletionRequestDto> decideDeletionRequest(
            @PathVariable Long id,
            @Valid @RequestBody DecideAccessRequestRequest body) {
        return ResponseEntity.ok(projectDeletionService.decide(id,
            AccessRequestStatus.valueOf(body.getDecision())));
    }
}
