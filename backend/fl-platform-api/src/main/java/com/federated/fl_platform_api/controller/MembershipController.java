package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.CreateMembershipRequest;
import com.federated.fl_platform_api.dto.MembershipDto;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.service.MembershipService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/projects/{projectId}/memberships")
public class MembershipController {

    @Autowired private MembershipService membershipService;

    @GetMapping
    public ResponseEntity<List<MembershipDto>> list(
            @PathVariable UUID projectId,
            @RequestParam(value = "role", required = false) String role) {
        MembershipRole filter = role == null ? null : MembershipRole.valueOf(role);
        return ResponseEntity.ok(membershipService.list(projectId, filter));
    }

    @PostMapping
    public ResponseEntity<MembershipDto> add(
            @PathVariable UUID projectId,
            @Valid @RequestBody CreateMembershipRequest body) {
        MembershipDto dto = membershipService.add(projectId, body.getUsername(),
            MembershipRole.valueOf(body.getRole()));
        return ResponseEntity.status(HttpStatus.CREATED).body(dto);
    }

    @DeleteMapping("/{userId}")
    public ResponseEntity<Void> remove(@PathVariable UUID projectId,
                                        @PathVariable Long userId) {
        membershipService.remove(projectId, userId);
        return ResponseEntity.noContent().build();
    }
}
