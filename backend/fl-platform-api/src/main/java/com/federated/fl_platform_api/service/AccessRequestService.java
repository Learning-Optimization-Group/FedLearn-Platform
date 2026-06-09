package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.*;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
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

@Service
public class AccessRequestService {

    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectAccessRequestRepository requestRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private AuthorizationService authz;
    @Autowired private NotificationService notifications;

    @Transactional
    public DecideAccessRequestResponse submit(UUID projectId, String message) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        // Org isolation (mutation): out-of-scope projects are a hard 403.
        authz.requireOrgScope(project.getOrgId());

        User caller = authz.currentUser();

        if (caller.getId().equals(project.getUser().getId())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "Owner cannot request access to their own project");
        }
        Optional<ProjectMembership> existing =
            membershipRepository.findByIdProjectIdAndIdUserId(projectId, caller.getId());
        if (existing.isPresent()
                && (existing.get().getRole() == MembershipRole.MEMBER
                 || existing.get().getRole() == MembershipRole.CLIENT)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Caller is already a participant in this project");
        }

        DecideAccessRequestResponse response = new DecideAccessRequestResponse();

        if (project.getVisibility() == ProjectVisibility.PUBLIC) {
            ProjectMembership m = new ProjectMembership(
                project, caller, MembershipRole.CLIENT, JoinedVia.PUBLIC_JOIN, caller);
            membershipRepository.save(m);
            response.setMembership(toMembershipDto(m));
            return response;
        }

        // PRIVATE: upsert a PENDING request. @PrePersist won't fire on UPDATE,
        // so we explicitly reset the lifecycle fields here.
        ProjectAccessRequest req = requestRepository
            .findByProjectIdAndUserId(projectId, caller.getId())
            .orElseGet(() -> new ProjectAccessRequest(project, caller, message));
        req.setMessage(message);
        req.setStatus(AccessRequestStatus.PENDING);
        req.setRequestedAt(Instant.now());
        req.setDecidedAt(null);
        req.setDecidedBy(null);
        ProjectAccessRequest saved = requestRepository.save(req);

        NotificationDto n = baseNotification(NotificationDto.Type.ACCESS_REQUEST_CREATED,
            project, caller, caller);
        notifications.notifyOwnerAndMembers(project.getId(), project.getUser().getId(), n);

        response.setRequest(toDto(saved));
        return response;
    }

    public List<AccessRequestDto> listForProject(UUID projectId, AccessRequestStatus filter) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        // Org isolation (read path): out-of-scope projects are treated as 404 so
        // cross-tenant project existence is not leaked.
        if (!authz.isInOrgScope(project.getOrgId())) {
            throw ResourceNotFoundException.project(projectId);
        }
        authz.requireOwnerOrMemberOrAdmin(project);

        List<ProjectAccessRequest> rows = (filter != null)
            ? requestRepository.findByProjectIdAndStatus(projectId, filter)
            : requestRepository.findByProjectId(projectId);
        return rows.stream().map(this::toDto).collect(Collectors.toList());
    }

    public List<AccessRequestDto> listMine() {
        User caller = authz.currentUser();
        return requestRepository.findByUserId(caller.getId())
            .stream().map(this::toDto).collect(Collectors.toList());
    }

    @Transactional
    public AccessRequestDto decide(UUID projectId, Long requestId, AccessRequestStatus decision) {
        if (decision != AccessRequestStatus.APPROVED && decision != AccessRequestStatus.DENIED) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "decision must be APPROVED or DENIED");
        }
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        // Org isolation (mutation): out-of-scope projects are a hard 403.
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrMemberOrAdmin(project);

        ProjectAccessRequest req = requestRepository.findById(requestId)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("AccessRequest", requestId));
        if (!req.getProject().getId().equals(projectId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "Request does not belong to this project");
        }
        if (req.getStatus() != AccessRequestStatus.PENDING) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Request has already been decided");
        }

        User actor = authz.currentUser();
        req.setStatus(decision);
        req.setDecidedAt(Instant.now());
        req.setDecidedBy(actor);
        ProjectAccessRequest saved = requestRepository.save(req);

        if (decision == AccessRequestStatus.APPROVED) {
            ProjectMembership m = new ProjectMembership(
                project, req.getUser(), MembershipRole.CLIENT,
                JoinedVia.REQUEST_APPROVED, actor);
            membershipRepository.save(m);
        }

        NotificationDto n = baseNotification(NotificationDto.Type.ACCESS_REQUEST_DECIDED,
            project, actor, req.getUser());
        n.setDecision(decision.name());
        notifications.notifyUser(req.getUser().getId(), n);

        return toDto(saved);
    }

    private NotificationDto baseNotification(NotificationDto.Type type,
                                              Project project, User actor, User subject) {
        NotificationDto n = new NotificationDto();
        n.setType(type);
        n.setProjectId(project.getId());
        n.setProjectName(project.getName());
        n.setActorId(actor.getId());
        n.setActorUsername(actor.getUsername());
        n.setSubjectId(subject.getId());
        n.setSubjectUsername(subject.getUsername());
        return n;
    }

    private AccessRequestDto toDto(ProjectAccessRequest r) {
        AccessRequestDto d = new AccessRequestDto();
        d.setId(r.getId());
        d.setProjectId(r.getProject().getId());
        d.setProjectName(r.getProject().getName());
        d.setUserId(r.getUser().getId());
        d.setUsername(r.getUser().getUsername());
        d.setStatus(r.getStatus().name());
        d.setMessage(r.getMessage());
        d.setRequestedAt(r.getRequestedAt());
        d.setDecidedAt(r.getDecidedAt());
        d.setDecidedByUsername(r.getDecidedBy() != null ? r.getDecidedBy().getUsername() : null);
        return d;
    }

    private MembershipDto toMembershipDto(ProjectMembership m) {
        MembershipDto d = new MembershipDto();
        d.setProjectId(m.getId().getProjectId());
        d.setUserId(m.getId().getUserId());
        d.setUsername(m.getUser() != null ? m.getUser().getUsername() : null);
        d.setRole(m.getRole().name());
        d.setPartitionId(m.getPartitionId());
        d.setJoinedVia(m.getJoinedVia() != null ? m.getJoinedVia().name() : null);
        d.setAddedAt(m.getAddedAt());
        return d;
    }
}
