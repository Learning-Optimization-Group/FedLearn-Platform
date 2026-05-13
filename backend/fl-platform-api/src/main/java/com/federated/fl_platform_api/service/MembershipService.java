package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.MembershipDto;
import com.federated.fl_platform_api.dto.NotificationDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class MembershipService {

    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private UserRepository userRepository;
    @Autowired private AuthorizationService authz;
    @Autowired private NotificationService notifications;

    public List<MembershipDto> list(UUID projectId, MembershipRole filterRole) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOwnerOrMemberOrAdmin(project);

        List<ProjectMembership> rows = (filterRole == null)
            ? membershipRepository.findByIdProjectId(projectId)
            : membershipRepository.findByIdProjectIdAndRole(projectId, filterRole);

        // Hide OWNER_SELF rows from API consumers (internal use only).
        return rows.stream()
            .filter(m -> m.getRole() != MembershipRole.OWNER)
            .map(this::toDto)
            .collect(Collectors.toList());
    }

    @Transactional
    public MembershipDto add(UUID projectId, String username, MembershipRole role) {
        if (role == MembershipRole.OWNER) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "OWNER memberships are not user-creatable");
        }
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));

        // Authorization: owner & admin can add any role. Member can only add CLIENT.
        if (authz.isAdmin() || authz.isOwner(project)) {
            // ok
        } else if (role == MembershipRole.CLIENT
                   && authz.hasMembership(project, MembershipRole.MEMBER)) {
            // ok
        } else {
            throw new AccessDeniedException("Not allowed to add this membership");
        }

        User target = userRepository.findByUsername(username)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", username));

        if (target.getId().equals(project.getUser().getId())) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Owner is implicitly a participant; do not create a membership row for them");
        }
        if (membershipRepository.findByIdProjectIdAndIdUserId(projectId, target.getId()).isPresent()) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "User is already a member of this project");
        }

        User addedBy = authz.currentUser();
        ProjectMembership m = new ProjectMembership(
            project, target, role, JoinedVia.OWNER_ADD, addedBy);
        ProjectMembership saved = membershipRepository.save(m);

        notifyMembershipChanged(NotificationDto.Type.MEMBERSHIP_ADDED,
            project, addedBy, target, role);
        return toDto(saved);
    }

    @Transactional
    public void remove(UUID projectId, Long userId) {
        Project project = projectRepository.findById(projectId)
            .orElseThrow(() -> ResourceNotFoundException.project(projectId));

        ProjectMembership existing = membershipRepository
            .findByIdProjectIdAndIdUserId(projectId, userId)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("Membership", userId));

        // Same authz rules as add(): owner/admin can remove any; member can remove CLIENT.
        if (authz.isAdmin() || authz.isOwner(project)) {
            // ok
        } else if (existing.getRole() == MembershipRole.CLIENT
                   && authz.hasMembership(project, MembershipRole.MEMBER)) {
            // ok
        } else {
            throw new AccessDeniedException("Not allowed to remove this membership");
        }

        membershipRepository.deleteByIdProjectIdAndIdUserId(projectId, userId);

        User actor = authz.currentUser();
        notifyMembershipChanged(NotificationDto.Type.MEMBERSHIP_REMOVED,
            project, actor, existing.getUser(), existing.getRole());
    }

    private void notifyMembershipChanged(NotificationDto.Type type,
                                          Project project, User actor, User subject,
                                          MembershipRole role) {
        NotificationDto n = new NotificationDto();
        n.setType(type);
        n.setProjectId(project.getId());
        n.setProjectName(project.getName());
        n.setActorId(actor.getId());
        n.setActorUsername(actor.getUsername());
        n.setSubjectId(subject.getId());
        n.setSubjectUsername(subject.getUsername());
        n.setRole(role.name());
        notifications.notifyUser(subject.getId(), n);
    }

    private MembershipDto toDto(ProjectMembership m) {
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
