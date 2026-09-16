package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ProjectStatusUpdateDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectInitStatus;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.repository.ProjectRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.*;

/**
 * BA-1: the async model-init worker. Called directly here (the {@code @Async} proxy is inert outside
 * a Spring context, so these run synchronously and deterministically) to verify the terminal
 * transitions: success -> DONE, failure -> FAILED, and each broadcasts the derived project status.
 */
@ExtendWith(MockitoExtension.class)
class ModelInitializationWorkerTest {

    @Mock ModelInitializer modelInitializer;
    @Mock ProjectRepository projectRepository;
    @Mock WebSocketService webSocketService;
    @Mock ProjectStatusService projectStatusService;

    @InjectMocks ModelInitializationWorker worker;

    private Project initializingProject() {
        Project p = new Project();
        p.setInitStatus(ProjectInitStatus.INITIALIZING);
        return p;
    }

    @Test
    void successfulInit_marksDone_andBroadcastsDerivedStatus() throws Exception {
        UUID pid = UUID.randomUUID();
        Project p = initializingProject();
        when(projectRepository.findById(pid)).thenReturn(Optional.of(p));
        when(projectStatusService.currentStatus(p)).thenReturn(ProjectStatus.CREATED);

        worker.initialize(pid, "CNN", "simple-cnn", "Adam", "/tmp/x.npz", 3, null);

        verify(modelInitializer).initializeModelFile("CNN", "simple-cnn", "Adam", "/tmp/x.npz", 3, null);
        assertEquals(ProjectInitStatus.DONE, p.getInitStatus());
        verify(projectRepository).save(p);

        ArgumentCaptor<ProjectStatusUpdateDto> cap = ArgumentCaptor.forClass(ProjectStatusUpdateDto.class);
        verify(webSocketService).sendStatusUpdate(cap.capture());
        assertEquals(pid, cap.getValue().getProjectId());
        assertEquals("CREATED", cap.getValue().getNewStatus());
    }

    @Test
    void failedInit_marksFailed_andBroadcastsFailed() throws Exception {
        UUID pid = UUID.randomUUID();
        Project p = initializingProject();
        doThrow(new IOException("python not found"))
                .when(modelInitializer).initializeModelFile(any(), any(), any(), any(), anyInt(), any());
        when(projectRepository.findById(pid)).thenReturn(Optional.of(p));
        when(projectStatusService.currentStatus(p)).thenReturn(ProjectStatus.FAILED);

        worker.initialize(pid, "CNN", "simple-cnn", "Adam", "/tmp/x.npz", 0, null);

        assertEquals(ProjectInitStatus.FAILED, p.getInitStatus());
        verify(projectRepository).save(p);
        verify(webSocketService).sendStatusUpdate(argThat(d -> "FAILED".equals(d.getNewStatus())));
    }

    @Test
    void projectDeletedMidInit_isANoOp() throws Exception {
        UUID pid = UUID.randomUUID();
        when(projectRepository.findById(pid)).thenReturn(Optional.empty());

        worker.initialize(pid, "CNN", "m", "Adam", "/tmp/x.npz", 0, null);

        verify(projectRepository, never()).save(any());
        verifyNoInteractions(webSocketService);
    }
}
