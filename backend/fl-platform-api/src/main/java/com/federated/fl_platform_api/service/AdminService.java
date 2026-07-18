package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.audit.Auditable;
import com.federated.fl_platform_api.dto.AdminOverviewDto;
import com.federated.fl_platform_api.dto.AdminUserDto;
import com.federated.fl_platform_api.dto.AuditEventDto;
import com.federated.fl_platform_api.dto.PagedResponseDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.OwnerPromotionRequestRepository;
import com.federated.fl_platform_api.repository.ProjectAccessRequestRepository;
import com.federated.fl_platform_api.repository.ProjectDeletionRequestRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class AdminService {

    /** Open upper bound for the audit-event search (see the repository note). */
    private static final Instant FAR_FUTURE = Instant.parse("9999-12-31T23:59:59Z");

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private OwnerPromotionRequestRepository ownerRequestRepository;
    @Autowired private ProjectDeletionRequestRepository deletionRequestRepository;
    @Autowired private ProjectAccessRequestRepository accessRequestRepository;
    @Autowired private AuditEventRepository auditEventRepository;
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
        return projectRepository.findAll().stream()
            .map(p -> toAdminProjectDto(p, projectStatusService.currentStatus(p)))   // BA-4
            .collect(Collectors.toList());
    }

    // ─── Search-first directories (server-side pagination) ───────────────────

    /**
     * Paginated users directory. {@code q} substring-matches username OR email
     * case-insensitively; {@code role}/{@code status} narrow further. Always
     * sorted username asc.
     */
    public PagedResponseDto<AdminUserDto> searchUsers(String q, String role, String status,
                                                      int page, int size) {
        PlatformRole roleFilter = parseEnumFilter(PlatformRole.class, role, "role");
        UserStatus statusFilter = parseEnumFilter(UserStatus.class, status, "status");
        Page<User> result = userRepository.searchForAdmin(
            likePattern(q), roleFilter, statusFilter,
            PageRequest.of(page, size, Sort.by(Sort.Direction.ASC, "username")));
        List<AdminUserDto> items = result.getContent().stream().map(this::toDto).toList();
        return new PagedResponseDto<>(items, page, size, result.getTotalElements());
    }

    /**
     * Paginated projects directory. {@code q} substring-matches project name OR
     * owner username case-insensitively. The {@code status} filter runs after
     * DB matching because project status is derived from the active run at read
     * time (BA-4) — it must be applied before slicing the page or {@code total}
     * would be wrong. Sorted name asc (from the repository query).
     */
    public PagedResponseDto<ProjectResponseDto> searchProjects(String q, String status, String visibility,
                                                               int page, int size) {
        ProjectStatus statusFilter = parseEnumFilter(ProjectStatus.class, status, "status");
        ProjectVisibility visibilityFilter = parseEnumFilter(ProjectVisibility.class, visibility, "visibility");
        List<Project> candidates = projectRepository.searchForAdmin(likePattern(q), visibilityFilter);

        List<Project> matched = new ArrayList<>();
        List<ProjectStatus> matchedStatuses = new ArrayList<>();
        for (Project p : candidates) {
            ProjectStatus derived = projectStatusService.currentStatus(p);
            if (statusFilter == null || derived == statusFilter) {
                matched.add(p);
                matchedStatuses.add(derived);
            }
        }

        long offset = (long) page * size;
        int fromIdx = (int) Math.min(offset, matched.size());
        int toIdx = (int) Math.min(offset + size, matched.size());
        List<ProjectResponseDto> items = new ArrayList<>();
        for (int i = fromIdx; i < toIdx; i++) {
            items.add(toAdminProjectDto(matched.get(i), matchedStatuses.get(i)));
        }
        return new PagedResponseDto<>(items, page, size, matched.size());
    }

    // ─── Account status (suspend / reactivate) ───────────────────────────────

    /**
     * Suspends an account. Two 409 guards mirror the last-admin demotion guard
     * in {@link #updateRole}: the platform must always keep at least one ACTIVE
     * PLATFORM_ADMIN, and an admin can never suspend their own account. Checked
     * in that order so the single-admin self-suspension case reports the more
     * specific "last active admin" conflict.
     */
    @Transactional
    @Auditable(action = AuditAction.USER_SUSPENDED, targetIdParam = "id", targetType = "USER")
    public AdminUserDto suspendUser(Long id) {
        User target = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));

        if (target.getPlatformRole() == PlatformRole.PLATFORM_ADMIN
                && target.getStatus() == UserStatus.ACTIVE) {
            long activeAdmins = userRepository.countByPlatformRoleAndStatus(
                PlatformRole.PLATFORM_ADMIN, UserStatus.ACTIVE);
            if (activeAdmins <= 1) {
                throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Cannot suspend the last active platform admin");
            }
        }
        if (isCurrentUser(target)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Admins cannot suspend their own account");
        }
        target.setStatus(UserStatus.SUSPENDED);
        return toDto(userRepository.save(target));
    }

    /** Reactivates a suspended (or pending) account back to ACTIVE. */
    @Transactional
    @Auditable(action = AuditAction.USER_REACTIVATED, targetIdParam = "id", targetType = "USER")
    public AdminUserDto reactivateUser(Long id) {
        User target = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));
        target.setStatus(UserStatus.ACTIVE);
        return toDto(userRepository.save(target));
    }

    // ─── Audit-event explorer ────────────────────────────────────────────────

    /**
     * Paginated audit-event search, newest first. {@code actor} is a username
     * resolved to the numeric actor id server-side; an unknown username matches
     * nothing (empty page, not an error). Actor usernames on the response are
     * resolved with one batched lookup per page — no N+1.
     */
    public PagedResponseDto<AuditEventDto> searchAuditEvents(String actor, String action, String targetType,
                                                             Instant from, Instant to, int page, int size) {
        Long actorId = null;
        if (actor != null && !actor.isBlank()) {
            Optional<User> actorUser = userRepository.findByUsername(actor.trim());
            if (actorUser.isEmpty()) {
                return new PagedResponseDto<>(List.of(), page, size, 0);
            }
            actorId = actorUser.get().getId();
        }
        AuditAction actionFilter = parseEnumFilter(AuditAction.class, action, "action");

        // The repository requires non-null bounds (a nullable Instant cannot be
        // bound through an IS-NULL guard on Postgres) — substitute sentinels
        // for the open ends.
        Page<AuditEvent> result = auditEventRepository.search(
            null, actorId, actionFilter, normalize(targetType),
            from != null ? from : Instant.EPOCH,
            to != null ? to : FAR_FUTURE,
            PageRequest.of(page, size));

        Set<Long> actorIds = result.getContent().stream()
            .map(AuditEvent::getActorUserId)
            .filter(Objects::nonNull)
            .collect(Collectors.toSet());
        Map<Long, String> usernamesById = actorIds.isEmpty() ? Map.of()
            : userRepository.findAllById(actorIds).stream()
                .collect(Collectors.toMap(User::getId, User::getUsername));

        List<AuditEventDto> items = result.getContent().stream()
            .map(e -> toAuditEventDto(e,
                e.getActorUserId() == null ? null : usernamesById.get(e.getActorUserId())))
            .toList();
        return new PagedResponseDto<>(items, page, size, result.getTotalElements());
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
        d.setStatus(u.getStatus() != null ? u.getStatus().name() : null);
        d.setDisplayName(u.getDisplayName());
        d.setLastLoginAt(u.getLastLoginAt());
        return d;
    }

    private ProjectResponseDto toAdminProjectDto(Project p, ProjectStatus status) {
        ProjectResponseDto d = new ProjectResponseDto();
        d.setId(p.getId());
        d.setName(p.getName());
        d.setModelType(p.getModelType());
        d.setModelName(p.getModelName());
        d.setServerPort(p.getServerPort());
        d.setOptimizer(p.getOptimizer());
        d.setStatus(status.name());   // BA-4: derived from the active run
        d.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
        d.setOwnerUsername(p.getUser() != null ? p.getUser().getUsername() : null);
        // Participants = MEMBER + CLIENT rows (exclude the internal OWNER_SELF
        // partition-holder row so the count reflects real collaborators).
        long participants = membershipRepository.findByIdProjectId(p.getId()).stream()
            .filter(m -> m.getRole() != MembershipRole.OWNER)
            .count();
        d.setParticipantCount((int) participants);
        return d;
    }

    private static AuditEventDto toAuditEventDto(AuditEvent e, String actorUsername) {
        AuditEventDto d = new AuditEventDto();
        d.setId(e.getId());
        d.setOccurredAt(e.getOccurredAt());
        d.setActorUserId(e.getActorUserId());
        d.setActorUsername(actorUsername);
        d.setAction(e.getAction() != null ? e.getAction().name() : null);
        d.setTargetType(e.getTargetType());
        d.setTargetId(e.getTargetId());
        d.setRequestIp(e.getRequestIp());
        d.setMetadata(e.getMetadata());
        return d;
    }

    /**
     * Parses an optional enum-valued filter. Blank/absent means "no filter";
     * an unknown value is a caller mistake — the IllegalArgumentException maps
     * to a 400 in GlobalExceptionHandler.
     */
    private static <E extends Enum<E>> E parseEnumFilter(Class<E> type, String raw, String paramName) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            return Enum.valueOf(type, raw.trim());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Invalid " + paramName + " filter: " + raw);
        }
    }

    private static String normalize(String s) {
        return (s == null || s.isBlank()) ? null : s.trim();
    }

    /**
     * Builds the non-null, pre-lowercased LIKE pattern the search queries
     * require ({@code "%"} = match-all when no q was given).
     */
    private static String likePattern(String q) {
        return (q == null || q.isBlank()) ? "%"
            : "%" + q.trim().toLowerCase(java.util.Locale.ROOT) + "%";
    }

    private static boolean isCurrentUser(User target) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        return auth != null && target.getUsername() != null
            && target.getUsername().equals(auth.getName());
    }
}
