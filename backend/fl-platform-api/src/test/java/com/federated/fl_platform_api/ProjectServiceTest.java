package com.federated.fl_platform_api;

import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ServerProcessException;
import com.federated.fl_platform_api.flower.FlowerServerManager;
import com.federated.fl_platform_api.service.ModelInitializer;
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

import java.io.IOException;
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
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@SuppressWarnings("null")
class ProjectServiceTest {

    @Mock
    private ProjectRepository projectRepository;

    @Mock
    private FlowerServerManager flowerServerManager;

    @Mock
    private ModelInitializer modelInitializer;

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
    private com.federated.fl_platform_api.service.ProjectStatusService projectStatusService;

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
    void whenCreateProject_thenShouldSucceedAndReturnProjectWithPortAndPath() throws Exception {
        // --- 1. ARRANGE ---
        when(authz.currentUser()).thenReturn(testUser);

        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
            Project p = invocation.getArgument(0);
            if (p.getId() == null) {
                p.setId(testProject.getId()); // Simulate ID generation on first save
            }
            return p;
        });

        doNothing().when(modelInitializer).initializeModelFile(anyString(), any(), any(), anyString(), anyInt(), any());

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

        verify(projectRepository, times(2)).save(any(Project.class));
        verify(modelInitializer, times(1)).initializeModelFile(eq(modelType), any(), any(), anyString(), eq(5), any());
    }

    @Test
    void whenModelInitializationFails_thenShouldThrowException() throws Exception {
        // --- ARRANGE ---
        when(authz.currentUser()).thenReturn(testUser);

        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
            Project p = invocation.getArgument(0);
            if (p.getId() == null) {
                p.setId(testProject.getId()); // Simulate ID generation on first save
            }
            return p;
        });

        doThrow(new IOException("Python script not found!"))
                .when(modelInitializer).initializeModelFile(anyString(), any(), any(), anyString(), anyInt(), any());

        CreateProjectRequest request = new CreateProjectRequest();
        request.setName(projectName);
        request.setModelType(modelType);
        request.setPretrainEpochs(5);

        // --- ACT & ASSERT ---
        // ProjectService now wraps IOException/InterruptedException coming out
        // of ModelInitializer in ServerProcessException so the @ControllerAdvice
        // can map it to a single 502 Bad Gateway response. The original cause
        // is preserved on the exception chain.
        ServerProcessException ex = assertThrows(ServerProcessException.class,
                () -> projectService.createProject(request));
        assertNotNull(ex.getCause());
        assertEquals(IOException.class, ex.getCause().getClass());
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
        m.setRole(com.federated.fl_platform_api.model.MembershipRole.CLIENT);
        when(membershipRepository.findByIdProjectIdAndIdUserId(joined.getId(), 42L))
            .thenReturn(java.util.Optional.of(m));

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
        lenient().when(runService.createForStart(any(), any(), anyInt(), anyInt(), anyInt())).thenReturn(run);

        // Model the FlowerServerManager runningServers map: a spawn flips it "running" and briefly
        // holds — widening the check-then-act window the race would otherwise exploit.
        AtomicBoolean running = new AtomicBoolean(false);
        AtomicInteger spawnCount = new AtomicInteger(0);
        lenient().when(flowerServerManager.isServerRunning(projectId)).thenAnswer(inv -> running.get());
        lenient().when(flowerServerManager.startServerForProject(any(), any(), anyInt(), anyInt()))
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
}
