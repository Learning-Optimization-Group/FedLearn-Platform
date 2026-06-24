package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ActiveRunDto;
import com.federated.fl_platform_api.dto.ClientConnectionDto;
import com.federated.fl_platform_api.dto.ClientProjectDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
public class ClientApiService {

    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private RunRepository runRepository;
    @Autowired private RunService runService;
    @Autowired private AuthorizationService authz;
    @Autowired private com.federated.fl_platform_api.security.OrgScope orgScope;

    @Value("${app.fl-server.grpc-host:localhost}")
    private String grpcHost;

    public List<ClientProjectDto> listForCurrentUser() {
        User self = authz.currentUser();
        List<Project> mine = orgScope.isUnrestricted()
            ? projectRepository.findOwnedOrMemberOf(self.getId())
            : projectRepository.findOwnedOrMemberOfInOrgs(self.getId(), orgScope.visibleOrgIds());
        List<ClientProjectDto> result = new ArrayList<>();
        for (Project p : mine) {
            boolean isOwner = p.getUser() != null && p.getUser().getId().equals(self.getId());
            boolean isClient = membershipRepository
                .existsByIdProjectIdAndIdUserIdAndRole(p.getId(), self.getId(), MembershipRole.CLIENT);
            if (isOwner || isClient) result.add(toDto(p, true));
        }
        List<Project> discoverable = orgScope.isUnrestricted()
            ? projectRepository.findDiscoverable(self.getId())
            : projectRepository.findDiscoverableInOrgs(self.getId(), orgScope.visibleOrgIds());
        for (Project p : discoverable) {
            if (p.getVisibility() == ProjectVisibility.PUBLIC) {
                result.add(toDto(p, false));
            }
        }
        return result;
    }

    public ClientProjectDto getOne(UUID projectId) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        if (!orgScope.allows(project.getOrgId())) {
            throw ResourceNotFoundException.project(projectId);
        }
        User self = authz.currentUser();
        boolean isOwner = project.getUser() != null && project.getUser().getId().equals(self.getId());
        boolean isClient = membershipRepository
            .existsByIdProjectIdAndIdUserIdAndRole(projectId, self.getId(), MembershipRole.CLIENT);
        boolean joined = isOwner || isClient;
        if (!joined && project.getVisibility() != ProjectVisibility.PUBLIC) {
            throw ResourceNotFoundException.project(projectId);
        }
        return toDto(project, joined);
    }

    @Transactional
    public ClientProjectDto join(UUID projectId) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        if (!orgScope.allows(project.getOrgId())) {
            throw ResourceNotFoundException.project(projectId);
        }
        User self = authz.currentUser();
        boolean isOwner = project.getUser() != null && project.getUser().getId().equals(self.getId());
        if (isOwner) return toDto(project, true);

        ProjectMembership existing = membershipRepository
            .findByIdProjectIdAndIdUserId(projectId, self.getId()).orElse(null);
        if (existing != null && existing.getRole() == MembershipRole.CLIENT) {
            return toDto(project, true);
        }
        switch (project.getVisibility()) {
            case PUBLIC -> {
                if (existing == null) {
                    membershipRepository.save(new ProjectMembership(
                        project, self, MembershipRole.CLIENT, JoinedVia.PUBLIC_JOIN, self));
                } else {
                    existing.setRole(MembershipRole.CLIENT);
                    membershipRepository.save(existing);
                }
                return toDto(project, true);
            }
            case RESTRICTED -> throw new AccessDeniedException(
                "This project requires an approved access request. Request access from the web app.");
            default -> throw ResourceNotFoundException.project(projectId);
        }
    }

    @Transactional
    public ClientConnectionDto getConnection(UUID projectId) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        if (project.getActiveRunId() == null) {
            throw new ProjectStateException(
                "Project is not currently running (status=" + project.getStatus() + ")");
        }
        // enroll enforces owner-or-CLIENT + run RUNNING and assigns the partition.
        com.federated.fl_platform_api.dto.EnrollmentDto enrollment =
            runService.enroll(project.getActiveRunId());

        ClientConnectionDto dto = new ClientConnectionDto();
        dto.setProjectId(projectId);
        dto.setName(project.getName());
        dto.setModelType(project.getModelType());
        dto.setServerAddress(enrollment.getGrpcEndpoint());
        dto.setPartitionId(enrollment.getPartitionId());
        dto.setStatus(project.getStatus());
        dto.setConnectionToken(enrollment.getConnectionToken());
        return dto;
    }

    private ClientProjectDto toDto(Project p, boolean joined) {
        ClientProjectDto d = new ClientProjectDto();
        d.setProjectId(p.getId());
        d.setName(p.getName());
        d.setModelType(p.getModelType());
        d.setRecipeKey(p.getModelType());
        d.setStatus(p.getStatus());
        d.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
        d.setJoined(joined);
        if (p.getActiveRunId() != null) {
            runRepository.findById(p.getActiveRunId()).ifPresent(r ->
                d.setActiveRun(new ActiveRunDto(r.getId(), r.getStatus().name())));
        }
        return d;
    }
}
