package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.AdminOverviewDto;
import com.federated.fl_platform_api.dto.AdminUserDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OwnerPromotionRequestRepository;
import com.federated.fl_platform_api.repository.ProjectAccessRequestRepository;
import com.federated.fl_platform_api.repository.ProjectDeletionRequestRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class AdminService {

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private OwnerPromotionRequestRepository ownerRequestRepository;
    @Autowired private ProjectDeletionRequestRepository deletionRequestRepository;
    @Autowired private ProjectAccessRequestRepository accessRequestRepository;
    @Autowired private ProjectStatusService projectStatusService;   // BA-4: derive status from the active run

    public List<AdminUserDto> listUsers() {
        return userRepository.findAll().stream().map(this::toDto).collect(Collectors.toList());
    }

    public AdminUserDto getUser(Long id) {
        User u = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));
        return toDto(u);
    }

    @Transactional
    public AdminUserDto updateRole(Long id, String newRole) {
        User target = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));

        PlatformRole role = PlatformRole.valueOf(newRole);

        if (role == PlatformRole.USER && target.getPlatformRole() == PlatformRole.PLATFORM_ADMIN) {
            long adminCount = userRepository.countByPlatformRole(PlatformRole.PLATFORM_ADMIN);
            if (adminCount <= 1) {
                throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Cannot demote the only remaining admin");
            }
        }
        target.setPlatformRole(role);
        return toDto(userRepository.save(target));
    }

    public List<ProjectResponseDto> listAllProjects() {
        return projectRepository.findAll().stream().map(p -> {
            ProjectResponseDto d = new ProjectResponseDto();
            d.setId(p.getId());
            d.setName(p.getName());
            d.setModelType(p.getModelType());
            d.setModelName(p.getModelName());
            d.setServerPort(p.getServerPort());
            d.setOptimizer(p.getOptimizer());
            d.setStatus(projectStatusService.currentStatus(p).name());   // BA-4
            d.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
            d.setOwnerUsername(p.getUser() != null ? p.getUser().getUsername() : null);
            // Participants = MEMBER + CLIENT rows (exclude the internal OWNER_SELF
            // partition-holder row so the count reflects real collaborators).
            long participants = membershipRepository.findByIdProjectId(p.getId()).stream()
                .filter(m -> m.getRole() != MembershipRole.OWNER)
                .count();
            d.setParticipantCount((int) participants);
            return d;
        }).collect(Collectors.toList());
    }

    /** Aggregate snapshot for the admin dashboard landing view. */
    public AdminOverviewDto getOverview() {
        AdminOverviewDto o = new AdminOverviewDto();
        o.setTotalUsers(userRepository.count());
        o.setOwners(userRepository.countByPlatformRole(PlatformRole.PROJECT_OWNER));
        o.setAdmins(userRepository.countByPlatformRole(PlatformRole.PLATFORM_ADMIN));
        o.setTotalProjects(projectRepository.count());
        // BA-4: derive from the active run so a project whose run FAILED is no longer over-counted
        // as running (the old projects.status string stayed "RUNNING" after a failed run).
        o.setRunningProjects(projectRepository.findAll().stream()
            .filter(p -> projectStatusService.currentStatus(p) == ProjectStatus.RUNNING).count());
        o.setPendingOwnerRequests(ownerRequestRepository.countByStatus(AccessRequestStatus.PENDING));
        o.setPendingDeletionRequests(deletionRequestRepository.countByStatus(AccessRequestStatus.PENDING));
        o.setPendingAccessRequests(accessRequestRepository.countByStatus(AccessRequestStatus.PENDING));
        return o;
    }

    private AdminUserDto toDto(User u) {
        AdminUserDto d = new AdminUserDto();
        d.setId(u.getId());
        d.setUsername(u.getUsername());
        d.setEmail(u.getEmail());
        d.setRole(u.getPlatformRole() != null ? u.getPlatformRole().name() : null);
        d.setProjectsOwned(projectRepository.findByUserId(u.getId()).size());
        d.setMemberships(membershipRepository.findByIdUserId(u.getId()).size());
        d.setCreatedAt(u.getCreatedAt());
        return d;
    }
}
