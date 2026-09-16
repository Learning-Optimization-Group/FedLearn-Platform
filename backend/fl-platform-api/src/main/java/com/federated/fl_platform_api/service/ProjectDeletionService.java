package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.DeletionRequestDto;
import com.federated.fl_platform_api.dto.NotificationDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectDeletionRequest;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectDeletionRequestRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Project-deletion workflow: an owner requests deletion (the FL server is
 * stopped and a PENDING request is filed); a platform admin then approves
 * (the project is hard-deleted) or denies. "Pending deletion" is represented
 * by the request row's existence, NOT by overloading {@code projects.status}
 * (that field drives the FL lifecycle: CREATED/RUNNING/STOPPED/COMPLETED).
 */
@Service
public class ProjectDeletionService {

    private static final Logger log = LoggerFactory.getLogger(ProjectDeletionService.class);

    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectDeletionRequestRepository requestRepository;
    @Autowired private ProjectService projectService;
    @Autowired private AuthorizationService authz;
    @Autowired private NotificationService notifications;

    @Transactional
    public DeletionRequestDto request(UUID projectId, String reason) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);

        Optional<ProjectDeletionRequest> existing = requestRepository.findByProjectId(projectId);
        if (existing.isPresent() && existing.get().getStatus() == AccessRequestStatus.PENDING) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "A deletion request for this project is already pending admin approval");
        }

        // Stop any running FL server so a project queued for deletion isn't still
        // training. Best-effort: a stop failure must not block filing the request.
        try {
            projectService.stopServerForProject(projectId);
        } catch (RuntimeException e) {
            log.warn("Failed to stop FL server for project {} during deletion request; continuing",
                projectId, e);
        }

        User caller = authz.currentUser();
        ProjectDeletionRequest req = existing
            .orElseGet(() -> new ProjectDeletionRequest(project, caller, reason));
        req.setRequestedBy(caller);
        req.setReason(reason);
        req.setStatus(AccessRequestStatus.PENDING);
        req.setRequestedAt(Instant.now());
        req.setDecidedAt(null);
        req.setDecidedBy(null);
        ProjectDeletionRequest saved = requestRepository.save(req);

        NotificationDto n = new NotificationDto();
        n.setType(NotificationDto.Type.PROJECT_DELETION_REQUESTED);
        n.setProjectId(project.getId());
        n.setProjectName(project.getName());
        n.setActorId(caller.getId());
        n.setActorUsername(caller.getUsername());
        notifications.notifyPlatformAdmins(n);

        return toDto(saved);
    }

    /** The deletion request for one project, if any (owner dashboard badge). */
    public Optional<DeletionRequestDto> getForProject(UUID projectId) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        if (!authz.isInOrgScope(project.getOrgId())) {
            throw ResourceNotFoundException.project(projectId);
        }
        authz.requireOwnerOrAdmin(project);
        return requestRepository.findByProjectId(projectId).map(this::toDto);
    }

    /** Admin queue. {@code filter} null returns all. */
    public List<DeletionRequestDto> listForAdmin(AccessRequestStatus filter) {
        authz.requirePlatformAdmin();
        List<ProjectDeletionRequest> rows = (filter != null)
            ? requestRepository.findByStatus(filter)
            : requestRepository.findAll();
        return rows.stream().map(this::toDto).collect(Collectors.toList());
    }

    @Transactional
    public DeletionRequestDto decide(Long requestId, AccessRequestStatus decision) {
        authz.requirePlatformAdmin();
        if (decision != AccessRequestStatus.APPROVED && decision != AccessRequestStatus.DENIED) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "decision must be APPROVED or DENIED");
        }
        ProjectDeletionRequest req = requestRepository.findById(requestId)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("ProjectDeletionRequest", requestId));
        if (req.getStatus() != AccessRequestStatus.PENDING) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Request has already been decided");
        }

        User actor = authz.currentUser();
        UUID projectId = req.getProject().getId();
        Long requesterId = req.getRequestedBy().getId();
        req.setStatus(decision);
        req.setDecidedAt(Instant.now());
        req.setDecidedBy(actor);

        // Build the response now: an APPROVED decision deletes this request row
        // (below) before returning, so we capture its fields while still attached.
        DeletionRequestDto dto = toDto(req);

        NotificationDto n = new NotificationDto();
        n.setType(NotificationDto.Type.PROJECT_DELETION_DECIDED);
        n.setProjectId(projectId);
        n.setProjectName(dto.getProjectName());
        n.setActorId(actor.getId());
        n.setActorUsername(actor.getUsername());
        n.setSubjectId(requesterId);
        n.setDecision(decision.name());
        notifications.notifyUser(requesterId, n);

        if (decision == AccessRequestStatus.APPROVED) {
            // Remove this request row first (and flush) so it doesn't dangle as an
            // FK to the project we're about to delete. The prod schema (V7) makes
            // this FK ON DELETE CASCADE, but the test schema is generated from the
            // entities, so we delete explicitly to be schema-independent. Then
            // hard-delete the project (admin context passes the owner-or-admin gate;
            // memberships/results/logs cascade via the schema's own FKs).
            requestRepository.delete(req);
            requestRepository.flush();
            projectService.deleteProject(projectId);
        } else {
            requestRepository.save(req);
        }
        return dto;
    }

    private DeletionRequestDto toDto(ProjectDeletionRequest r) {
        DeletionRequestDto d = new DeletionRequestDto();
        d.setId(r.getId());
        d.setProjectId(r.getProject().getId());
        d.setProjectName(r.getProject().getName());
        d.setRequestedById(r.getRequestedBy().getId());
        d.setRequestedByUsername(r.getRequestedBy().getUsername());
        d.setStatus(r.getStatus().name());
        d.setReason(r.getReason());
        d.setRequestedAt(r.getRequestedAt());
        d.setDecidedAt(r.getDecidedAt());
        d.setDecidedByUsername(r.getDecidedBy() != null ? r.getDecidedBy().getUsername() : null);
        return d;
    }
}
