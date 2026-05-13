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
import com.federated.fl_platform_api.service.WebSocketService;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.util.UUID;

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

    @InjectMocks
    private ProjectService projectService;

    private Project testProject;
    private String projectName;
    private String modelType;
    private User testUser;

    @BeforeEach
    void setUp() {
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

        doNothing().when(modelInitializer).initializeModelFile(anyString(), any(), any(), anyString(), anyInt());

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
        verify(modelInitializer, times(1)).initializeModelFile(eq(modelType), any(), any(), anyString(), eq(5));
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
                .when(modelInitializer).initializeModelFile(anyString(), any(), any(), anyString(), anyInt());

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
}