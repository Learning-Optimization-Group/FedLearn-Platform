
package com.federated.fl_platform_api;

import com.federated.fl_platform_api.flower.FlowerServerManager;
//import com.federated.fl_platform_api.initializer.ModelInitializer;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.service.ProjectService;
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
import static org.mockito.Mockito.*;

// This annotation integrates Mockito with the JUnit 5 test lifecycle.
@ExtendWith(MockitoExtension.class)
class ProjectServiceTest {

    // @Mock creates a fake version of this class. Calls to it will be intercepted.
    @Mock
    private ProjectRepository projectRepository;

    @Mock
    private FlowerServerManager flowerServerManager;

    @Mock
//    private ModelInitializer modelInitializer;

    // @InjectMocks creates an instance of ProjectService and injects the mocked
    // dependencies (@Mock annotated fields) into it.
    @InjectMocks
    private ProjectService projectService;

    private Project testProject;
    private String projectName;
    private String modelType;

    // This method runs before each @Test method. It's great for setting up common objects.
    @BeforeEach
    void setUp() {
        projectName = "My Test CNN Project";
        modelType = "CNN";

        testProject = new Project();
        testProject.setId(UUID.randomUUID());
        testProject.setName(projectName);
        testProject.setModelType(modelType);
    }

//    @Test
//    void whenCreateProject_thenShouldSucceedAndReturnProjectWithPortAndPath() throws Exception {
//        // --- 1. ARRANGE (Define the behavior of our mocks) ---
//
//        // When projectRepository.save() is called, we want to simulate the database
//        // saving the object and returning it, possibly with a generated ID.
//        // We use thenAnswer to handle the two separate calls to save().
//        when(projectRepository.save(any(Project.class))).thenAnswer(invocation -> {
//            Project p = invocation.getArgument(0);
//            if (p.getId() == null) {
//                p.setId(testProject.getId()); // Simulate ID generation on first save
//            }
//            return p; // Return the saved/updated project
//        });
//
//        // When modelInitializer.initializeModelFile() is called, we don't want to
//        // actually run the Python script. We just want to confirm it was called.
//        // doNothing() is perfect for void methods.
//        doNothing().when(modelInitializer).initializeModelFile(anyString(), anyString());
//
//        // When flowerServerManager.startServerForProject() is called, we simulate
//        // it successfully finding and returning a free port.
//        int expectedPort = 9090;
//        when(flowerServerManager.startServerForProject(any(Project.class))).thenReturn(expectedPort);
//
//
//        // --- 2. ACT (Execute the method we are testing) ---
//        Project createdProject = projectService.createProject(projectName, modelType);
//
//
//        // --- 3. ASSERT (Verify the results and interactions) ---
//
//        // Assert that the returned object has the correct data
//        assertNotNull(createdProject);
//        assertEquals(testProject.getId(), createdProject.getId());
//        assertEquals(projectName, createdProject.getName());
//        assertEquals(expectedPort, createdProject.getServerPort());
//        assertTrue(createdProject.getModelPath().contains(testProject.getId().toString()));
//
//        // Verify that our mocked methods were called with the correct parameters
//        // and the correct number of times.
//
//        // We expect save() to be called twice: once to get an ID, once to update with port/path.
//        verify(projectRepository, times(2)).save(any(Project.class));
//
//        // Verify the model initializer was called once with the correct model type and a valid path.
//        String expectedModelPath = "models/" + testProject.getId().toString() + ".flwr";
//        verify(modelInitializer, times(1)).initializeModelFile(modelType, expectedModelPath);
//
//        // Verify the flower server manager was called once.
//        verify(flowerServerManager, times(1)).startServerForProject(any(Project.class));
//    }
//
//    // ProjectServiceTest.java
//
//    @Test
//    void whenModelInitializationFails_thenShouldThrowException() throws Exception {
//        // --- ARRANGE ---
//
//        // FIX: Add this missing mock setup.
//        // We still need to simulate the repository returning a valid project,
//        // otherwise the code will fail before it even gets to the part we want to test.
//        when(projectRepository.save(any(Project.class))).thenReturn(testProject);
//
//        // This part was already correct. We are telling the initializer to fail.
//        doThrow(new IOException("Python script not found!"))
//                .when(modelInitializer).initializeModelFile(anyString(), anyString());
//
//        // --- ACT & ASSERT ---
//
//        // Now, the test should behave as expected. createProject will successfully get past
//        // the repository save, then fail exactly where we want it to.
//        assertThrows(IOException.class, () -> {
//            projectService.createProject(projectName, modelType);
//        });
//
//        // This verification is now more meaningful because we know the code didn't
//        // crash prematurely. We are confirming that because of the IOException,
//        // the flower server was never started.
//        verify(flowerServerManager, never()).startServerForProject(any(Project.class));
//    }
}