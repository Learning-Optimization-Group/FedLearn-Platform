package com.federated.fl_platform_api;

// Imports from your own project packages
import com.federated.fl_platform_api.config.SecurityConfig; // Important: keep this import
import com.federated.fl_platform_api.controller.ProjectController;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.CustomUserDetailsService; // Important: import the class to be mocked
import com.federated.fl_platform_api.service.ProjectService;

// Imports from external libraries
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import; // Important: keep this import
import org.springframework.http.MediaType;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import java.util.UUID;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(controllers = ProjectController.class)
@Import(SecurityConfig.class) // This is correct, we want to test with our real security rules.
public class ProjectControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;
//
//    // We already have this mock because the controller depends on it.
//    @MockBean
//    private ProjectService projectService;
//
//    // --- THIS IS THE FIX ---
//    // Because we imported SecurityConfig, and SecurityConfig needs a CustomUserDetailsService,
//    // we must provide a mock for it.
//    @MockBean
//    private CustomUserDetailsService customUserDetailsService;
//
//
//    @Test
//    @WithMockUser
//    void whenPostCreateProject_withValidRequest_thenReturns200AndProjectDetails() throws Exception {
//        // --- ARRANGE ---
//        ProjectController.CreateProjectRequest request = new ProjectController.CreateProjectRequest();
//        request.name = "Final WebMvc Test";
//        request.modelType = "CNN";
//
//        Project mockProjectResponse = new Project();
//        mockProjectResponse.setId(UUID.randomUUID());
//        mockProjectResponse.setName(request.name);
//        mockProjectResponse.setModelType(request.modelType);
//        mockProjectResponse.setServerPort(9090);
//        mockProjectResponse.setModelPath("models/" + mockProjectResponse.getId().toString() + ".flwr");
//
//        // The mock for projectService remains the same.
//        when(projectService.createProject(anyString(), anyString())).thenReturn(mockProjectResponse);
//
//        // We don't need to define any behavior for customUserDetailsService mock.
//        // Its mere existence is enough to satisfy the dependency and allow the context to load.
//
//        // --- ACT ---
//        ResultActions resultActions = mockMvc.perform(post("/api/projects")
//                .contentType(MediaType.APPLICATION_JSON)
//                .content(objectMapper.writeValueAsString(request)));
//
//        // --- ASSERT ---
//        resultActions
//                .andExpect(status().isOk())
//                .andExpect(jsonPath("$.id").value(mockProjectResponse.getId().toString()))
//                .andExpect(jsonPath("$.name").value(request.name));
//    }
}