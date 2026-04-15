package com.federated.fl_platform_api;

import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.flower.FlowerServerManager;
import com.federated.fl_platform_api.service.ModelInitializer;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.ProjectService;
import com.federated.fl_platform_api.service.WebSocketService;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;

import java.io.IOException;
import java.util.Optional;
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
    private UserRepository userRepository;

    @Mock
    private FlowerServerManager flowerServerManager;

    @Mock
    private ModelInitializer modelInitializer;

    @Mock
    private RoundResultRepository roundResultRepository;

    @Mock
    private WebSocketService webSocketService;

    @Mock
    private SecurityContext securityContext;

    @Mock
    private Authentication authentication;

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
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getName()).thenReturn("testuser");
        SecurityContextHolder.setContext(securityContext);
        
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.of(testUser));

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
        when(securityContext.getAuthentication()).thenReturn(authentication);
        when(authentication.getName()).thenReturn("testuser");
        SecurityContextHolder.setContext(securityContext);
        
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.of(testUser));

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
        assertThrows(IOException.class, () -> {
            projectService.createProject(request);
        });
    }
}