package com.federated.fl_platform_api;

import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.flower.FlowerServerManager;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.ServerLogRepository;
import com.federated.fl_platform_api.service.ModelInitializer;
import com.federated.fl_platform_api.service.ProjectService;
import com.federated.fl_platform_api.service.WebSocketService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.PageRequest;
import org.springframework.security.access.AccessDeniedException;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@SuppressWarnings("null")
class ProjectServiceExtendedTest {

    @Mock private ProjectRepository projectRepository;
    @Mock private FlowerServerManager flowerServerManager;
    @Mock private ModelInitializer modelInitializer;
    @Mock private RoundResultRepository roundResultRepository;
    @Mock private WebSocketService webSocketService;
    @Mock private ServerLogRepository serverLogRepository;
    @Mock private com.federated.fl_platform_api.service.AuthorizationService authz;
    @Mock private com.federated.fl_platform_api.repository.ProjectMembershipRepository membershipRepository;

    @InjectMocks
    private ProjectService projectService;

    private User testUser;
    private Project testProject;

    @BeforeEach
    void setUp() {
        testUser = new User();
        testUser.setId(1L);
        testUser.setUsername("testuser");

        testProject = new Project();
        testProject.setId(UUID.randomUUID());
        testProject.setName("Test Project");
        testProject.setModelType("CNN");
        testProject.setModelName("resnet18");
        testProject.setOptimizer("SGD");
        testProject.setUser(testUser);
        testProject.setStatus("STOPPED");
    }

    // Helper kept for legacy callers — auth checks are now centralised in
    // AuthorizationService, which we mock as a no-op for the happy path.
    private void asRegularUser() {
        // Default mock behaviour for void methods is no-op, which is what we
        // want for the "caller is permitted" path.
    }

    @Test
    void getProjectsForCurrentUser_shouldReturnOnlyCallerProjects() {
        Project p1 = new Project();
        p1.setId(UUID.randomUUID());
        p1.setName("P1"); p1.setModelType("CNN"); p1.setModelName("r"); p1.setOptimizer("SGD");
        p1.setUser(testUser); p1.setStatus("CREATED");

        when(authz.currentUser()).thenReturn(testUser);
        when(projectRepository.findOwnedOrMemberOf(1L)).thenReturn(List.of(p1));

        List<ProjectResponseDto> results = projectService.getProjectsForCurrentUser();

        assertEquals(1, results.size());
        assertEquals("P1", results.get(0).getName());
        assertEquals("OWNER", results.get(0).getMyRelationship());
    }

    @Test
    void stopServerForProject_whenRunning_shouldSetStatusToStopped() {
        testProject.setStatus("RUNNING");
        when(projectRepository.findById(testProject.getId())).thenReturn(Optional.of(testProject));
        asRegularUser();
        when(flowerServerManager.stopServerForProject(testProject.getId())).thenReturn(true);
        when(projectRepository.save(any(Project.class))).thenAnswer(inv -> inv.getArgument(0));

        ProjectResponseDto dto = projectService.stopServerForProject(testProject.getId());

        assertEquals("STOPPED", dto.getStatus());
        verify(projectRepository).save(any(Project.class));
    }

    @Test
    void deleteProject_shouldStopServerThenDelete() {
        when(projectRepository.findById(testProject.getId())).thenReturn(Optional.of(testProject));
        asRegularUser();
        when(flowerServerManager.stopServerForProject(testProject.getId())).thenReturn(true);

        projectService.deleteProject(testProject.getId());

        verify(flowerServerManager).stopServerForProject(testProject.getId());
        verify(projectRepository).deleteById(testProject.getId());
    }

    @Test
    void startServerForProject_whenAlreadyRunning_shouldThrowProjectStateException() {
        testProject.setStatus("RUNNING");
        when(projectRepository.findById(testProject.getId())).thenReturn(Optional.of(testProject));
        asRegularUser();
        when(flowerServerManager.isServerRunning(testProject.getId())).thenReturn(true);

        assertThrows(ProjectStateException.class,
                () -> projectService.startServerForProject(testProject.getId(), null));
    }

    @Test
    void getLogsForProject_shouldClampPageSizeToMax() {
        when(projectRepository.findById(testProject.getId())).thenReturn(Optional.of(testProject));
        asRegularUser();
        when(serverLogRepository.findByProjectIdOrderByTimestampAsc(any(), any()))
                .thenReturn(Collections.emptyList());

        // Request a page size larger than the allowed max (500)
        projectService.getLogsForProject(testProject.getId(), PageRequest.of(0, 999999));

        // Verify the repo was called with a clamped page size
        verify(serverLogRepository).findByProjectIdOrderByTimestampAsc(
                eq(testProject.getId()),
                argThat(p -> p.getPageSize() <= ProjectService.MAX_LOGS_PAGE_SIZE)
        );
    }

    @Test
    void stopServerForProject_whenCallerIsNotOwner_shouldThrowAccessDeniedException() {
        when(projectRepository.findById(testProject.getId())).thenReturn(Optional.of(testProject));
        // AuthorizationService rejects the caller; ProjectService must surface
        // the AccessDeniedException unchanged.
        doThrow(new AccessDeniedException("You do not have access to this project"))
                .when(authz).requireOwnerOrAdmin(testProject);

        assertThrows(AccessDeniedException.class,
                () -> projectService.stopServerForProject(testProject.getId()));
    }
}
