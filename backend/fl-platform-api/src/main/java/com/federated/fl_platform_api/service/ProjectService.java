package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.audit.Auditable;
import com.federated.fl_platform_api.dto.*;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.exception.ServerProcessException;
import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectAccessRequestRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.flower.FlowerServerManager;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class ProjectService {

    private static final Logger log = LoggerFactory.getLogger(ProjectService.class);

    @Autowired
    private ProjectRepository projectRepository;
    @Autowired
    private FlowerServerManager flowerServerManager;
    @Autowired
    private ModelInitializer modelInitializer;
    @Autowired
    private ProjectMembershipRepository membershipRepository;
    @Autowired
    private RoundResultRepository roundResultRepository;
    @Autowired
    private WebSocketService webSocketService;
    @Autowired
    private com.federated.fl_platform_api.repository.ServerLogRepository serverLogRepository;
    @Autowired
    private AuthorizationService authz;
    @Autowired
    private ProjectAccessRequestRepository accessRequestRepository;
    @Autowired
    private NotificationService notificationService;
    @Autowired
    private OrganizationMembershipRepository orgMembershipRepository;
    @Autowired
    private com.federated.fl_platform_api.security.OrgScope orgScope;

    /**
     * Default org UUID seeded by the V5 migration — the single transitional
     * bootstrap org. Used as the fallback both for project creation and for the
     * OrgScope of users that have no explicit org membership yet. Exposed so
     * {@link com.federated.fl_platform_api.security.OrgScopeFilter} shares the
     * exact same default (no duplicate UUID literal across the codebase).
     */
    public static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");


    private RoundResultDto convertToDto(RoundResult result) {
        RoundResultDto dto = new RoundResultDto();
        dto.setId(result.getId());
        dto.setServerRound(result.getServerRound());
        dto.setLoss(result.getLoss());
        dto.setAccuracy(result.getAccuracy());
        dto.setGpuUtilization(result.getGpuUtilization());
        return dto;
    }

    private ProjectResponseDto convertToDto(Project project) {
        ProjectResponseDto dto = new ProjectResponseDto();
        dto.setId(project.getId());
        dto.setName(project.getName());
        dto.setModelType(project.getModelType());
        dto.setModelName(project.getModelName());
        dto.setServerPort(project.getServerPort());
        dto.setOptimizer(project.getOptimizer());
        dto.setStatus(project.getStatus());
        dto.setVisibility(project.getVisibility() != null ? project.getVisibility().name() : null);

        return dto;
    }

    @Transactional
    @SuppressWarnings("null")
    @Auditable(action = AuditAction.PROJECT_CREATED, targetType = "PROJECT")
    public ProjectResponseDto createProject(CreateProjectRequest request) throws IOException, InterruptedException {
        log.info("Creating project '{}' (modelType={})", request.getName(), request.getModelType());

        User owner = authz.currentUser();

        Project project = new Project();
        project.setName(request.getName());
        project.setModelType(request.getModelType());
        project.setModelName(request.getModelName());
        project.setOptimizer(request.getOptimizer());
        project.setUser(owner);
        // V5 made projects.org_id NOT NULL. Pin the project to the owner's first
        // org membership; fall back to the Default org (seeded by V5) for users
        // that somehow have no membership. Real cross-org selection UI lives in
        // a later sub-spec.
        UUID orgId = orgMembershipRepository.findByUserId(owner.getId()).stream()
                .findFirst()
                .map(m -> m.getOrgId())
                .orElse(DEFAULT_ORG_ID);
        project.setOrgId(orgId);
        project.setStatus("CREATED");
        Project savedProject = projectRepository.save(project);
        log.debug("Persisted project shell with id={}", savedProject.getId());

        File modelFile = new File("models/" + savedProject.getId().toString() + ".npz");
        if (!modelFile.getParentFile().exists() && !modelFile.getParentFile().mkdirs()) {
            throw new ServerProcessException(
                    "Could not create model output directory: " + modelFile.getParentFile().getAbsolutePath());
        }
        String absoluteModelPath = modelFile.getAbsolutePath();
        savedProject.setModelPath(absoluteModelPath);

        try {
            modelInitializer.initializeModelFile(
                    request.getModelType(),
                    request.getModelName(),
                    request.getOptimizer(),
                    absoluteModelPath,
                    request.getPretrainEpochs());
        } catch (IOException | InterruptedException e) {
            // Allow the @Transactional rollback to drop the orphan project row.
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new ServerProcessException(
                    "Model initialization failed for project " + savedProject.getId(), e);
        }

        Project finalProject = projectRepository.save(savedProject);
        log.info("Project {} fully initialised at {}", finalProject.getId(), absoluteModelPath);
        return convertToDto(finalProject);
    }

    @Auditable(action = AuditAction.RUN_STARTED, targetIdParam = "projectId", targetType = "PROJECT")
    public ProjectResponseDto startServerForProject(@NonNull UUID projectId, StartProject request)
            throws IOException, InterruptedException {

        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);

        String strategyToUse = (request != null && request.getStrategy() != null && !request.getStrategy().isEmpty())
                ? request.getStrategy()
                : "FedAvg";

        Integer minClients = (request != null && request.getMinClients() != null)
                ? request.getMinClients()
                : 1;

        Integer numRoundsToUse;
        if (request != null && request.getNumRounds() != null && request.getNumRounds() > 0) {
            numRoundsToUse = request.getNumRounds();
        } else {
            // Default for both LLMs and CNNs; tune per-model-type if needed.
            numRoundsToUse = 5;
        }
        log.debug("Starting project {} with strategy={}, rounds={}, minClients={}",
                projectId, strategyToUse, numRoundsToUse, minClients);

        if (flowerServerManager.isServerRunning(projectId)) {
            // Was previously a silent fall-through that double-spawned the server.
            throw new ProjectStateException(
                    "FL server is already running for project " + projectId
                            + " on port " + project.getServerPort());
        }

        Optional<Integer> port = flowerServerManager.startServerForProject(
                project, strategyToUse, numRoundsToUse, minClients);
        project.setServerPort(port.orElse(null));
        project.setStatus("RUNNING");

        Project updatedProject = projectRepository.save(project);

        ProjectStatusUpdateDto update = new ProjectStatusUpdateDto(
                updatedProject.getId(), "RUNNING", updatedProject.getServerPort());
        webSocketService.sendStatusUpdate(update);
        if (port.isPresent()) {
            log.info("Started FL server for project {} on port {}", projectId, port.get());
        } else {
            log.info("Started FL server for project {} on ECS (port managed externally)", projectId);
        }

        return convertToDto(updatedProject);
    }

    @Transactional
    @Auditable(action = AuditAction.RUN_STOPPED, targetIdParam = "projectId", targetType = "PROJECT")
    public ProjectResponseDto stopServerForProject(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);

        boolean stopped = flowerServerManager.stopServerForProject(projectId);
        Project finalProjectState = project;
        if (stopped || "RUNNING".equals(project.getStatus())) {
            project.setServerPort(null);
            project.setStatus("STOPPED");
            finalProjectState = projectRepository.save(project);
            log.info("Stopped FL server for project {}", projectId);
        } else {
            log.debug("No running server found for project {}; nothing to stop", projectId);
        }

        return convertToDto(finalProjectState);
    }

    public List<ProjectResponseDto> getProjectsForCurrentUser() {
        User caller = authz.currentUser();
        // Platform admins (unrestricted scope) see all orgs via the unscoped
        // query; everyone else is constrained to their visible orgs (which falls
        // back to the single default org for membership-less users).
        List<Project> projects = orgScope.isUnrestricted()
                ? projectRepository.findOwnedOrMemberOf(caller.getId())
                : projectRepository.findOwnedOrMemberOfInOrgs(
                        caller.getId(), orgScope.visibleOrgIds());
        return projects.stream().map(p -> {
            ProjectResponseDto dto = convertToDto(p);
            dto.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
            if (p.getUser() != null && p.getUser().getId().equals(caller.getId())) {
                dto.setMyRelationship("OWNER");
            } else {
                ProjectMembership m = membershipRepository
                        .findByIdProjectIdAndIdUserId(p.getId(), caller.getId())
                        .orElse(null);
                dto.setMyRelationship(m != null && m.getRole() != null ? m.getRole().name() : null);
            }
            return dto;
        }).collect(Collectors.toList());
    }

    public List<RoundResultDto> getResultsForProject(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);
        return roundResultRepository.findByProjectIdOrderByServerRoundAsc(projectId).stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
    }

    @Transactional
    public void markProjectAsCompleted(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));

        project.setStatus("COMPLETED");
        project.setServerPort(null);
        projectRepository.save(project);

        webSocketService.sendStatusUpdate(
                new ProjectStatusUpdateDto(project.getId(), "COMPLETED", null));
        log.info("Project {} marked as completed", projectId);
    }

    @Transactional
    @Auditable(action = AuditAction.PROJECT_DELETED, targetIdParam = "projectId", targetType = "PROJECT")
    public void deleteProject(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);

        // Best-effort: stop any running FL server before removing the row so
        // we don't leak processes/ECS tasks.
        try {
            flowerServerManager.stopServerForProject(projectId);
        } catch (RuntimeException e) {
            log.warn("Failed to stop FL server for project {} before delete; continuing",
                    projectId, e);
        }
        projectRepository.deleteById(projectId);
        log.info("Project {} deleted", projectId);
    }

    /** Hard cap on the page size a caller can request for the live log endpoint. */
    public static final int MAX_LOGS_PAGE_SIZE = 500;
    /** Default page size when the caller doesn't specify one. */
    public static final int DEFAULT_LOGS_PAGE_SIZE = 200;
    /**
     * Hard cap on the export endpoint. Larger projects need to fall back to
     * archived storage (S3 Athena, etc.) — not in scope yet.
     */
    public static final int MAX_LOGS_EXPORT_SIZE = 10_000;

    public List<ServerLogDto> getLogsForProject(@NonNull UUID projectId, Pageable requested) {
        Project project = requireProjectAndOwnership(projectId);

        // Clamp to MAX_LOGS_PAGE_SIZE so a caller can't ask for an unbounded
        // result by passing ?size=1000000. Sort is server-controlled so the
        // caller can't reverse the order or sort by an unindexed column.
        int pageNumber = requested != null ? Math.max(0, requested.getPageNumber()) : 0;
        int pageSize = requested != null && requested.getPageSize() > 0
                ? Math.min(requested.getPageSize(), MAX_LOGS_PAGE_SIZE)
                : DEFAULT_LOGS_PAGE_SIZE;
        Pageable safePageable = PageRequest.of(pageNumber, pageSize, Sort.by("timestamp").ascending());

        return serverLogRepository.findByProjectIdOrderByTimestampAsc(project.getId(), safePageable)
                .stream()
                .map(ProjectService::toLogDto)
                .collect(Collectors.toList());
    }

    /**
     * Returns up to {@link #MAX_LOGS_EXPORT_SIZE} log lines for the export
     * endpoint. The cap protects the JVM from a runaway project; if real
     * users need bigger exports we should ship them to S3 instead.
     */
    public List<ServerLogDto> getLogsForExport(@NonNull UUID projectId) {
        Project project = requireProjectAndOwnership(projectId);
        Pageable cap = PageRequest.of(0, MAX_LOGS_EXPORT_SIZE, Sort.by("timestamp").ascending());
        return serverLogRepository.findByProjectIdOrderByTimestampAsc(project.getId(), cap)
                .stream()
                .map(ProjectService::toLogDto)
                .collect(Collectors.toList());
    }

    private Project requireProjectAndOwnership(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);
        return project;
    }

    private static ServerLogDto toLogDto(com.federated.fl_platform_api.model.ServerLog entry) {
        ServerLogDto dto = new ServerLogDto();
        dto.setLevel(entry.getLevel());
        dto.setMessage(entry.getMessage());
        dto.setStackTrace(entry.getStackTrace());
        dto.setTimestamp(entry.getTimestamp());
        return dto;
    }

    @Transactional
    @SuppressWarnings("null")
    public ProjectResponseDto updateProject(@NonNull UUID projectId, UpdateProjectRequest req) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        authz.requireOrgScope(project.getOrgId());
        authz.requireOwnerOrAdmin(project);

        if (req.getName() != null && !req.getName().isBlank()) {
            project.setName(req.getName());
        }
        if (req.getDescription() != null) {
            project.setModelDescription(req.getDescription());
        }
        if (req.getVisibility() != null) {
            ProjectVisibility next = ProjectVisibility.valueOf(req.getVisibility());
            if (project.getVisibility() != next) {
                project.setVisibility(next);
                // Notify current participants (excluding internal OWNER_SELF rows).
                User actor = authz.currentUser();
                NotificationDto n = new NotificationDto();
                n.setType(NotificationDto.Type.PROJECT_VISIBILITY_CHANGED);
                n.setProjectId(project.getId());
                n.setProjectName(project.getName());
                n.setActorId(actor.getId());
                n.setActorUsername(actor.getUsername());
                for (ProjectMembership m : membershipRepository.findByIdProjectId(project.getId())) {
                    if (m.getRole() != MembershipRole.OWNER) {
                        notificationService.notifyUser(m.getId().getUserId(), n);
                    }
                }
            }
        }
        Project saved = projectRepository.save(project);
        ProjectResponseDto dto = convertToDto(saved);
        if (authz.isOwner(saved)) {
            dto.setMyRelationship("OWNER");
        }
        return dto;
    }

    public ProjectResponseDto getProject(@NonNull UUID projectId) {
        Project project = projectRepository.findById(projectId)
                .orElseThrow(() -> ResourceNotFoundException.project(projectId));
        // Org isolation: a project outside the caller's visible orgs is treated
        // as non-existent (404) so we don't leak cross-tenant existence. Platform
        // admins are unrestricted and skip this gate.
        if (!orgScope.allows(project.getOrgId())) {
            throw ResourceNotFoundException.project(projectId);
        }
        boolean isAdmin = authz.isPlatformAdmin();
        boolean isOwner = authz.isOwner(project);
        boolean isParticipant = isAdmin || isOwner
                || authz.myMembership(project).map(m ->
                      m.getRole() == MembershipRole.MEMBER
                   || m.getRole() == MembershipRole.CLIENT).orElse(false);

        if (isParticipant) {
            ProjectResponseDto dto = convertToDto(project);
            if (isOwner) {
                dto.setMyRelationship("OWNER");
            } else if (!isAdmin) {
                authz.myMembership(project)
                        .ifPresent(m -> dto.setMyRelationship(m.getRole().name()));
            }
            return dto;
        }

        if (project.getVisibility() == ProjectVisibility.PUBLIC) {
            // Outsiders only see the world-readable fields of a PUBLIC project.
            ProjectResponseDto trimmed = new ProjectResponseDto();
            trimmed.setId(project.getId());
            trimmed.setName(project.getName());
            trimmed.setModelType(project.getModelType());
            trimmed.setStatus(project.getStatus());
            trimmed.setVisibility("PUBLIC");
            return trimmed;
        }
        // PRIVATE outsiders get 404 so we don't leak existence.
        throw ResourceNotFoundException.project(projectId);
    }

    public List<DiscoverProjectDto> getDiscoverProjects() {
        User caller = authz.currentUser();
        List<Project> candidates = orgScope.isUnrestricted()
                ? projectRepository.findDiscoverable(caller.getId(), ProjectVisibility.PUBLIC)
                : projectRepository.findDiscoverableInOrgs(
                        caller.getId(), ProjectVisibility.PUBLIC, orgScope.visibleOrgIds());
        return candidates
                .stream()
                .filter(p -> p.getUser() == null || !p.getUser().getId().equals(caller.getId()))
                .filter(p -> membershipRepository
                        .findByIdProjectIdAndIdUserId(p.getId(), caller.getId())
                        .map(m -> m.getRole() != MembershipRole.MEMBER
                                  && m.getRole() != MembershipRole.CLIENT)
                        .orElse(true))
                .map(p -> toDiscoverDto(p, caller.getId()))
                .collect(Collectors.toList());
    }

    private DiscoverProjectDto toDiscoverDto(Project p, Long callerId) {
        DiscoverProjectDto d = new DiscoverProjectDto();
        d.setId(p.getId());
        d.setName(p.getName());
        d.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
        d.setOwnerUsername(p.getUser() != null ? p.getUser().getUsername() : null);
        d.setModelType(p.getModelType());
        d.setDescription(p.getModelDescription());
        d.setMyRequestStatus(accessRequestRepository
                .findByProjectIdAndUserId(p.getId(), callerId)
                .map(r -> r.getStatus().name())
                .orElse("NONE"));
        return d;
    }
}
