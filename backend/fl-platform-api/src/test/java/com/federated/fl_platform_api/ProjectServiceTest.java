package com.federated.fl_platform_api;

import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.StartProject;
import com.federated.fl_platform_api.model.RobustAggregationSettings;
import com.federated.fl_platform_api.model.RobustMethod;
import org.mockito.ArgumentCaptor;
import com.federated.fl_platform_api.orchestration.FlServerManager;
import com.federated.fl_platform_api.service.ModelInitializationWorker;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.service.ProjectService;
import com.federated.fl_platform_api.service.RunService;
import com.federated.fl_platform_api.service.WebSocketService;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.exception.ProjectStateException;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@SuppressWarnings("null")
class ProjectServiceTest {

    @Mock
    private ProjectRepository projectRepository;

    @Mock
    private FlServerManager flServerManager;

    @Mock
    private ModelInitializationWorker modelInitWorker;

    @Mock
    private RoundResultRepository roundResultRepository;

    @Mock
    private WebSocketService webSocketService;

    @Mock
    private com.federated.fl_platform_api.service.AuthorizationService authz;

    @Mock
    private com.federated.fl_platform_api.repository.ProjectMembershipRepository membershipRepository;

    @Mock
    private com.federated.fl_platform_api.repository.OrganizationMembershipRepository orgMembershipRepository;

    @Mock
    private com.federated.fl_platform_api.security.OrgScope orgScope;

    @Mock
    private RunService runService;

    @Mock
    private com.federated.fl_platform_api.service.ModelBundleStager modelBundleStager;

    @Mock
    private com.federated.fl_platform_api.service.ProjectStatusService projectStatusService;

    @Mock
    private com.federated.fl_platform_api.service.RegistryModelResolver registryModelResolver;

    @InjectMocks
    private ProjectService projectService;

    private Project testProject;
    private String projectName;
    private String modelType;
    private User testUser;

    @BeforeEach
    void setUp() {
        // BA-4: project status is now derived. Mock the deriver as the identity on the project's
        // stored status string so the existing status assertions still describe the same behavior;
        // the real run->status derivation is covered by ProjectStatusServiceTest.
        org.mockito.Mockito.lenient().when(projectStatusService.currentStatus(org.mockito.ArgumentMatchers.any()))
            .thenAnswer(inv -> {
                String s = ((Project) inv.getArgument(0)).getStatus();
                return s == null ? com.federated.fl_platform_api.model.ProjectStatus.CREATED
                                 : com.federated.fl_platform_api.model.ProjectStatus.valueOf(s);
            });
        projectName = "My Test CNN Project";
        modelType = "CNN";

        testUser = new User();
        testUser.setUsername("testuser");

        testProject = new Project();
        testProject.setId(UUID.randomUUID());
        testProject.setName(projectName);
        testProject.setModelType(modelType);
        testProject.setUser(testUser);
    }

    @Test
    void whenCreateProject_thenPersistsShellAndDispatchesAsyncInit() throws Exception {
        // --- 1. ARRANGE ---
        when(authz.currentUser()).thenReturn(testUser);

        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
            Project p = invocation.getArgument(0);
            if (p.getId() == null) {
                p.setId(testProject.getId()); // Simulate ID generation on first save
            }
            return p;
        });

        CreateProjectRequest request = new CreateProjectRequest();
        request.setName(projectName);
        request.setModelType(modelType);
        request.setPretrainEpochs(5);

        // --- 2. ACT ---
        ProjectResponseDto createdProject = projectService.createProject(request);

        // --- 3. ASSERT ---
        assertNotNull(createdProject);
        assertEquals(testProject.getId(), createdProject.getId());
        assertEquals(projectName, createdProject.getName());

        // BA-1: model init no longer runs inline in the request. The shell (+ model path) is persisted
        // in two saves, then the Python-spawning init is dispatched to the async worker. There is no
        // ambient transaction in this Mockito unit test, so the dispatch fires immediately rather than
        // after-commit — verifying the worker is invoked exactly once with the project's parameters.
        verify(projectRepository, times(2)).save(any(Project.class));
        verify(modelInitWorker, times(1))
                .initialize(eq(testProject.getId()), eq(modelType), any(), any(), anyString(), eq(5), any());
        // The init failure path (formerly a synchronous throw here) is now the worker's concern and is
        // covered by ModelInitializationWorkerTest.failedInit_marksFailed_andBroadcastsFailed.
    }

    @Test
    void getProjectsForCurrentUser_returnsOwnedAndMemberOfWithRelationship() {
        when(authz.currentUser()).thenReturn(testUser);
        testUser.setId(42L);

        Project owned = new Project();
        owned.setId(UUID.randomUUID());
        owned.setName("owned-by-me");
        owned.setUser(testUser);
        owned.setVisibility(com.federated.fl_platform_api.model.ProjectVisibility.PRIVATE);

        User otherOwner = new User();
        otherOwner.setId(7L);
        otherOwner.setUsername("someone-else");

        Project joined = new Project();
        joined.setId(UUID.randomUUID());
        joined.setName("joined-as-client");
        joined.setUser(otherOwner);
        joined.setVisibility(com.federated.fl_platform_api.model.ProjectVisibility.PUBLIC);

        // Unrestricted scope routes to the unscoped query (org-scoped filtering
        // is covered separately in OrgIsolationTest).
        when(orgScope.isUnrestricted()).thenReturn(true);
        when(projectRepository.findOwnedOrMemberOf(42L))
            .thenReturn(java.util.List.of(owned, joined));

        com.federated.fl_platform_api.model.ProjectMembership m =
            new com.federated.fl_platform_api.model.ProjectMembership();
        m.setId(new com.federated.fl_platform_api.model.ProjectMembershipId(joined.getId(), 42L));
        m.setRole(com.federated.fl_platform_api.model.MembershipRole.CLIENT);
        // BA-10: the dashboard list now batches the caller's memberships in one
        // query (findByIdUserIdAndIdProjectIdIn) instead of one lookup per project.
        when(membershipRepository.findByIdUserIdAndIdProjectIdIn(eq(42L), any()))
            .thenReturn(java.util.List.of(m));

        java.util.List<ProjectResponseDto> dtos = projectService.getProjectsForCurrentUser();

        assertEquals(2, dtos.size());
        ProjectResponseDto o = dtos.stream().filter(d -> "owned-by-me".equals(d.getName())).findFirst().orElseThrow();
        assertEquals("OWNER",  o.getMyRelationship());
        assertEquals("PRIVATE", o.getVisibility());
        ProjectResponseDto j = dtos.stream().filter(d -> "joined-as-client".equals(d.getName())).findFirst().orElseThrow();
        assertEquals("CLIENT", j.getMyRelationship());
        assertEquals("PUBLIC", j.getVisibility());
    }

    // BA-2: concurrent /start for one project must spawn EXACTLY ONE FL server. The check-then-act
    // (isServerRunning -> spawn) race previously let N callers all pass the running-check before any
    // spawn registered, then all spawn — duplicate servers, one orphaned/untracked process.
    @Test
    void concurrentStart_spawnsExactlyOneServer_losersGet409() throws Exception {
        UUID projectId = testProject.getId();
        testProject.setStatus("CREATED");

        lenient().when(projectRepository.findById(projectId)).thenReturn(Optional.of(testProject));
        lenient().when(projectRepository.save(any(Project.class))).thenAnswer(inv -> inv.getArgument(0));

        Run run = mock(Run.class);
        lenient().when(run.getId()).thenReturn(UUID.randomUUID());
        lenient().when(runService.createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any())).thenReturn(run);

        // Model the FlServerManager runningServers map: a spawn flips it "running" and briefly
        // holds — widening the check-then-act window the race would otherwise exploit.
        AtomicBoolean running = new AtomicBoolean(false);
        AtomicInteger spawnCount = new AtomicInteger(0);
        lenient().when(flServerManager.isServerRunning(projectId)).thenAnswer(inv -> running.get());
        lenient().when(flServerManager.startServerForProject(any(), any(), anyInt(), anyInt(), any(), any()))
            .thenAnswer(inv -> {
                spawnCount.incrementAndGet();
                Thread.sleep(60);
                running.set(true);
                return Optional.of(50000);
            });

        int n = 8;
        ExecutorService pool = Executors.newFixedThreadPool(n);
        CountDownLatch ready = new CountDownLatch(n);
        CountDownLatch go = new CountDownLatch(1);
        AtomicInteger successes = new AtomicInteger(0);
        AtomicInteger conflicts = new AtomicInteger(0);
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            futures.add(pool.submit(() -> {
                ready.countDown();
                try {
                    go.await();
                    projectService.startServerForProject(projectId, null);
                    successes.incrementAndGet();
                } catch (ProjectStateException e) {
                    conflicts.incrementAndGet();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }));
        }
        assertTrue(ready.await(5, TimeUnit.SECONDS), "threads did not arm in time");
        go.countDown();  // release all n callers at once — they hit the running-check together
        for (Future<?> f : futures) f.get(15, TimeUnit.SECONDS);
        pool.shutdownNow();

        assertEquals(1, spawnCount.get(), "exactly one FL server spawned (no double-spawn)");
        assertEquals(1, successes.get(), "exactly one caller succeeds");
        assertEquals(n - 1, conflicts.get(), "losers get a deterministic 409 (ProjectStateException)");
    }

    // MO-15: a successful start auto-stages this run's on-device bundle with its recipe (the project's
    // model type), so a mobile client that joins finds it at GET /api/runs/{runId}/model-bundle.
    private Run arrangeSingleStart() {
        UUID projectId = testProject.getId();
        testProject.setStatus("CREATED");
        lenient().when(projectRepository.findById(projectId)).thenReturn(Optional.of(testProject));
        lenient().when(projectRepository.save(any(Project.class))).thenAnswer(inv -> inv.getArgument(0));
        Run run = mock(Run.class);
        lenient().when(run.getId()).thenReturn(UUID.randomUUID());
        lenient().when(runService.createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any())).thenReturn(run);
        lenient().when(flServerManager.isServerRunning(projectId)).thenReturn(false);
        lenient().when(flServerManager.startServerForProject(any(), any(), anyInt(), anyInt(), any(), any()))
                .thenReturn(Optional.of(50000));
        return run;
    }

    @Test
    void startServerForProject_autoStagesBundleWithRunIdAndRecipe() throws Exception {
        Run run = arrangeSingleStart();
        projectService.startServerForProject(testProject.getId(), null);
        // staged with THIS run's id and the project's model type (recipe key)
        verify(modelBundleStager).stageForRun(run.getId(), modelType);
    }

    @Test
    void startServerForProject_stagerFailureDoesNotFailTheStart() {
        arrangeSingleStart();
        // A misbehaving stager (contract says never-throw, but be defensive) must not fail the start:
        // the bundle is phone-only, so staging failure degrades to a graceful 404, never a start failure.
        doThrow(new RuntimeException("boom")).when(modelBundleStager).stageForRun(any(), any());
        assertDoesNotThrow(() -> projectService.startServerForProject(testProject.getId(), null));
        // the FL server still spawned despite the staging failure
        verify(flServerManager).startServerForProject(any(), any(), anyInt(), anyInt(), any(), any());
    }

    // ─── SE-11: DP policy at project creation ────────────────────────────────────────────────────
    // A regulated (or DP-enabled) project must carry a complete DP config at creation:
    // dpTargetEpsilon > 0, dpDelta in (0,1) exclusive, dpClipNorm > 0.

    private CreateProjectRequest dpRequest(Boolean regulated, Boolean dpEnabled,
                                           Double epsilon, Double delta, Double clipNorm) {
        CreateProjectRequest r = new CreateProjectRequest();
        r.setName(projectName);
        r.setModelType(modelType);
        r.setPretrainEpochs(5);
        r.setRegulated(regulated);
        r.setDpEnabled(dpEnabled);
        r.setDpTargetEpsilon(epsilon);
        r.setDpDelta(delta);
        r.setDpClipNorm(clipNorm);
        return r;
    }

    @Test
    void createRegulatedProject_withoutDpConfig_isRejectedBeforePersisting() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(true, false, null, null, null)));
        // The message must be actionable: name the three knobs and the guidance range.
        assertTrue(ex.getMessage().contains("dpTargetEpsilon"), ex.getMessage());
        assertTrue(ex.getMessage().contains("dpDelta"), ex.getMessage());
        assertTrue(ex.getMessage().contains("dpClipNorm"), ex.getMessage());
        assertTrue(ex.getMessage().contains("4-8"), "message should carry the epsilon guidance range: " + ex.getMessage());
        verify(projectRepository, never()).save(any(Project.class));
        verify(modelInitWorker, never()).initialize(any(), any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    void createDpEnabledProject_withIncompleteConfig_isRejected() {
        // clip norm missing
        assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(false, true, 6.0, 1e-5, null)));
        verify(projectRepository, never()).save(any(Project.class));
    }

    @Test
    void createDpProject_withNonPositiveEpsilonOrClipNorm_isRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(false, true, 0.0, 1e-5, 1.5)));
        assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(false, true, 6.0, 1e-5, 0.0)));
        verify(projectRepository, never()).save(any(Project.class));
    }

    @Test
    void createDpProject_withDeltaOutsideOpenUnitInterval_isRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(false, true, 6.0, 0.0, 1.5)));
        assertThrows(IllegalArgumentException.class,
                () -> projectService.createProject(dpRequest(false, true, 6.0, 1.0, 1.5)));
        verify(projectRepository, never()).save(any(Project.class));
    }

    @Test
    void createRegulatedDpProject_withCompleteConfig_persistsThePolicyFields() throws Exception {
        when(authz.currentUser()).thenReturn(testUser);
        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
            Project p = invocation.getArgument(0);
            if (p.getId() == null) {
                p.setId(testProject.getId());
            }
            return p;
        });

        projectService.createProject(dpRequest(true, true, 6.0, 1e-5, 1.5));

        org.mockito.ArgumentCaptor<Project> saved = org.mockito.ArgumentCaptor.forClass(Project.class);
        verify(projectRepository, times(2)).save(saved.capture());
        Project persisted = saved.getValue();
        assertTrue(persisted.isRegulated());
        assertTrue(persisted.isDpEnabled());
        assertEquals(6.0, persisted.getDpTargetEpsilon());
        assertEquals(1e-5, persisted.getDpDelta());
        assertEquals(1.5, persisted.getDpClipNorm());
    }

    @Test
    void createPlainProject_defaultsDpPolicyOff() throws Exception {
        when(authz.currentUser()).thenReturn(testUser);
        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
            Project p = invocation.getArgument(0);
            if (p.getId() == null) {
                p.setId(testProject.getId());
            }
            return p;
        });

        CreateProjectRequest request = new CreateProjectRequest();
        request.setName(projectName);
        request.setModelType(modelType);
        request.setPretrainEpochs(5);

        projectService.createProject(request);

        org.mockito.ArgumentCaptor<Project> saved = org.mockito.ArgumentCaptor.forClass(Project.class);
        verify(projectRepository, times(2)).save(saved.capture());
        Project persisted = saved.getValue();
        assertFalse(persisted.isRegulated());
        assertFalse(persisted.isDpEnabled());
        assertNull(persisted.getDpTargetEpsilon());
        assertNull(persisted.getDpDelta());
        assertNull(persisted.getDpClipNorm());
    }

    // ---- BA-11: inference resolves the model from the registry (via RegistryModelResolver) ----

    private Project inferableProject(UUID orgId, String modelType) {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setOrgId(orgId);
        p.setModelType(modelType);
        p.setModelName("net");
        p.setStatus("COMPLETED");
        return p;
    }

    @Test
    void resolveInferenceTarget_servesTheRegistryPath_whenTheResolverResolvesOne(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws Exception {
        UUID orgId = UUID.randomUUID();
        Project p = inferableProject(orgId, "CNN");
        p.setModelPath(dir.resolve("stale.npz").toString()); // present, but the registry must win
        java.nio.file.Path registryFile = dir.resolve("deadbeef.npz");
        java.nio.file.Files.write(registryFile, new byte[]{4, 2});
        when(projectRepository.findById(p.getId())).thenReturn(Optional.of(p));
        when(orgScope.allows(orgId)).thenReturn(true);
        when(registryModelResolver.resolveModelPath(p)).thenReturn(Optional.of(registryFile.toString()));

        var target = projectService.resolveInferenceTarget(p.getId());

        assertEquals(registryFile.toString(), target.modelPath()); // the registry path, not the .npz
    }

    @Test
    void resolveInferenceTarget_fallsBackToNpz_whenTheResolverIsEmpty(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws Exception {
        UUID orgId = UUID.randomUUID();
        Project p = inferableProject(orgId, "CNN");
        java.nio.file.Path npz = dir.resolve("model.npz");
        java.nio.file.Files.write(npz, new byte[]{1, 2, 3});
        p.setModelPath(npz.toString());
        when(projectRepository.findById(p.getId())).thenReturn(Optional.of(p));
        when(orgScope.allows(orgId)).thenReturn(true);
        when(registryModelResolver.resolveModelPath(p)).thenReturn(Optional.empty());

        var target = projectService.resolveInferenceTarget(p.getId());

        assertEquals(npz.toString(), target.modelPath());
    }

    @Test
    void resolveInferenceTarget_409_whenNeitherRegistryNorNpz() {
        UUID orgId = UUID.randomUUID();
        Project p = inferableProject(orgId, "CNN"); // no modelPath set
        when(projectRepository.findById(p.getId())).thenReturn(Optional.of(p));
        when(orgScope.allows(orgId)).thenReturn(true);
        when(registryModelResolver.resolveModelPath(p)).thenReturn(Optional.empty());

        assertThrows(
                com.federated.fl_platform_api.exception.ProjectStateException.class,
                () -> projectService.resolveInferenceTarget(p.getId()));
    }

    // ─── Robust aggregation settings at /start ──────────────────────────────────────────────────
    // Validated against the RESOLVED request (strategy and minClients after defaults), then the one
    // settings object is both persisted on the run and passed to the spawn, so the run record and the
    // argv cannot disagree about which rule ran.

    private void arrangeRobustStart() {
        UUID projectId = testProject.getId();
        testProject.setStatus("CREATED");
        lenient().when(projectRepository.findById(projectId)).thenReturn(Optional.of(testProject));
        lenient().when(projectRepository.save(any(Project.class))).thenAnswer(inv -> inv.getArgument(0));
        Run run = mock(Run.class);
        lenient().when(run.getId()).thenReturn(UUID.randomUUID());
        lenient().when(runService.createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any())).thenReturn(run);
        lenient().when(flServerManager.isServerRunning(projectId)).thenReturn(false);
        lenient().when(flServerManager.startServerForProject(any(), any(), anyInt(), anyInt(), any(), any()))
                .thenReturn(Optional.of(50000));
    }

    private static StartProject startRequest(String strategy, String method, Double fraction, Integer minClients) {
        StartProject r = new StartProject();
        r.setStrategy(strategy);
        r.setRobustMethod(method);
        r.setByzantineFraction(fraction);
        r.setMinClients(minClients);
        return r;
    }

    private void assertNothingCreatedOrSpawned() {
        verify(runService, never()).createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any());
        verify(flServerManager, never()).startServerForProject(any(), any(), anyInt(), anyInt(), any(), any());
    }

    @Test
    void robustStart_feasibleRule_persistsAndSpawnsTheSameSettings() throws Exception {
        arrangeRobustStart();
        projectService.startServerForProject(testProject.getId(), startRequest("Robust", "BULYAN", 0.2, 20));

        RobustAggregationSettings expected = new RobustAggregationSettings(RobustMethod.BULYAN, 0.2, null, null);
        ArgumentCaptor<RobustAggregationSettings> persisted = ArgumentCaptor.forClass(RobustAggregationSettings.class);
        ArgumentCaptor<RobustAggregationSettings> spawned = ArgumentCaptor.forClass(RobustAggregationSettings.class);
        verify(runService).createForStart(eq(testProject), eq("Robust"), anyInt(), eq(20), anyInt(), persisted.capture(), isNull());
        verify(flServerManager).startServerForProject(eq(testProject), eq("Robust"), anyInt(), eq(20), spawned.capture(), isNull());
        assertEquals(expected, persisted.getValue());
        assertEquals(expected, spawned.getValue());
    }

    @Test
    void robustStart_ruleThatCannotRunAtMinClients_isRefusedBeforeAnythingIsCreated() {
        arrangeRobustStart();
        // f = int(0.2 * 10) = 2 and Bulyan needs 4*2+3 = 11 > 10: the server would refuse every round.
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), startRequest("Robust", "BULYAN", 0.2, 10)));
        assertTrue(ex.getMessage().contains("BULYAN"), ex.getMessage());
        assertTrue(ex.getMessage().contains("minClients = 10"), ex.getMessage());
        assertNothingCreatedOrSpawned();
    }

    @Test
    void robustStart_krumAtTheStartDialogDefaultOfTwoClients_isRefused() {
        arrangeRobustStart();
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), startRequest("Robust", "KRUM", 0.0, 2)));
        assertTrue(ex.getMessage().contains("at least 3"), ex.getMessage());
        assertNothingCreatedOrSpawned();
    }

    @Test
    void robustFieldsOnAnotherStrategy_areRefusedRatherThanIgnored() {
        arrangeRobustStart();
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), startRequest("FedAvg", "MEDIAN", null, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), startRequest("FedAvg", null, 0.1, 2)));
        assertNothingCreatedOrSpawned();
    }

    @Test
    void robustParametersWithoutAMethod_areRefused() {
        arrangeRobustStart();
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), startRequest("Robust", null, 0.1, 20)));
        assertTrue(ex.getMessage().contains("robustMethod"), ex.getMessage());
        assertNothingCreatedOrSpawned();
    }

    @Test
    void aParameterTheChosenRuleDoesNotRead_isRefused() {
        arrangeRobustStart();
        StartProject krumWithTrim = startRequest("Robust", "KRUM", 0.1, 20);
        krumWithTrim.setTrimRatio(0.2);
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), krumWithTrim));

        StartProject medianWithTau = startRequest("Robust", "MEDIAN", null, 20);
        medianWithTau.setCenteredClipTau(1.0);
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), medianWithTau));
        assertNothingCreatedOrSpawned();
    }

    @Test
    void robustStartWithoutSettings_passesNoneSoTheServerDefaultAndArgvAreUnchanged() throws Exception {
        arrangeRobustStart();
        projectService.startServerForProject(testProject.getId(), startRequest("Robust", null, null, 2));
        verify(runService).createForStart(eq(testProject), eq("Robust"), anyInt(), eq(2), anyInt(), isNull(), isNull());
        verify(flServerManager).startServerForProject(eq(testProject), eq("Robust"), anyInt(), eq(2), isNull(), isNull());
    }

    // ─── Secure aggregation at /start ──────────────────────────────────────────────────────────
    // Offered only where it does something (DeComFL), only where it can work (client auth on), and only with a
    // threshold a round can meet (<= minClients, which is exactly what every DeComFL round aggregates). The one
    // threshold is persisted and passed to the spawn.

    private void arrangeSecureStart(boolean clientAuthRequired) {
        UUID projectId = testProject.getId();
        testProject.setStatus("CREATED");
        lenient().when(projectRepository.findById(projectId)).thenReturn(Optional.of(testProject));
        lenient().when(projectRepository.save(any(Project.class))).thenAnswer(inv -> inv.getArgument(0));
        Run run = mock(Run.class);
        lenient().when(run.getId()).thenReturn(UUID.randomUUID());
        lenient().when(runService.createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any())).thenReturn(run);
        lenient().when(flServerManager.isServerRunning(projectId)).thenReturn(false);
        lenient().when(flServerManager.isClientAuthRequired()).thenReturn(clientAuthRequired);
        lenient().when(flServerManager.startServerForProject(any(), any(), anyInt(), anyInt(), any(), any()))
                .thenReturn(Optional.of(50000));
    }

    private static StartProject secureStart(String strategy, Boolean secure, Integer threshold, Integer minClients) {
        StartProject r = new StartProject();
        r.setStrategy(strategy);
        r.setSecureAggregation(secure);
        r.setSecureAggThreshold(threshold);
        r.setMinClients(minClients);
        return r;
    }

    private void assertNoSecureRunCreatedOrSpawned() {
        verify(runService, never()).createForStart(any(), any(), anyInt(), anyInt(), anyInt(), any(), any());
        verify(flServerManager, never()).startServerForProject(any(), any(), anyInt(), anyInt(), any(), any());
    }

    @Test
    void secureStart_onDeComFL_persistsAndSpawnsTheSameThreshold() throws Exception {
        arrangeSecureStart(true);
        projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", true, 3, 5));
        verify(runService).createForStart(eq(testProject), eq("DeComFL"), anyInt(), eq(5), anyInt(), isNull(), eq(3));
        verify(flServerManager).startServerForProject(eq(testProject), eq("DeComFL"), anyInt(), eq(5), isNull(), eq(3));
    }

    @Test
    void secureStart_withoutAThreshold_usesTheServerDefaultOfTwo() throws Exception {
        arrangeSecureStart(true);
        projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", true, null, 2));
        verify(flServerManager).startServerForProject(eq(testProject), eq("DeComFL"), anyInt(), eq(2), isNull(), eq(2));
    }

    @Test
    void secureStart_onAnotherStrategy_isRefusedBecauseTheFlagWouldDoNothing() {
        arrangeSecureStart(true);
        for (String strategy : List.of("FedAvg", "Robust", "FedProx")) {
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                    projectService.startServerForProject(testProject.getId(), secureStart(strategy, true, 2, 5)));
            assertTrue(ex.getMessage().contains("DeComFL"), ex.getMessage());
        }
        assertNoSecureRunCreatedOrSpawned();
    }

    @Test
    void secureStart_withoutClientAuth_isRefusedBecauseEveryRoundWouldFail() {
        arrangeSecureStart(false);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", true, 2, 5)));
        assertTrue(ex.getMessage().toLowerCase().contains("client auth"), ex.getMessage());
        assertNoSecureRunCreatedOrSpawned();
    }

    @Test
    void secureStart_thresholdAboveMinClients_isRefusedBecauseNoRoundCouldMeetIt() {
        arrangeSecureStart(true);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", true, 4, 3)));
        assertTrue(ex.getMessage().contains("minClients = 3"), ex.getMessage());
        assertNoSecureRunCreatedOrSpawned();
    }

    @Test
    void aThresholdWithoutSecureAggregationOn_isRefused() {
        arrangeSecureStart(true);
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", null, 2, 5)));
        assertThrows(IllegalArgumentException.class, () ->
                projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", false, 2, 5)));
        assertNoSecureRunCreatedOrSpawned();
    }

    @Test
    void plainDeComFLStart_carriesNoSecureAggregationAndNeedsNoClientAuth() throws Exception {
        arrangeSecureStart(false);
        projectService.startServerForProject(testProject.getId(), secureStart("DeComFL", false, null, 2));
        verify(runService).createForStart(eq(testProject), eq("DeComFL"), anyInt(), eq(2), anyInt(), isNull(), isNull());
        verify(flServerManager).startServerForProject(eq(testProject), eq("DeComFL"), anyInt(), eq(2), isNull(), isNull());
    }
}
