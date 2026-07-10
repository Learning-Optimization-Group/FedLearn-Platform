package com.federated.fl_platform_api.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.dto.UpdateProjectRequest;
import com.federated.fl_platform_api.exception.GlobalExceptionHandler;
import com.federated.fl_platform_api.service.ProjectDeletionService;
import com.federated.fl_platform_api.service.ProjectService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.http.converter.json.Jackson2ObjectMapperBuilder;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * BA-15: web-layer contract for the {@code visibility} field of
 * {@code PATCH /api/projects/{id}}. The live bug was a bean-validation regression —
 * the field was validated with a hardcoded {@code @Pattern(regexp = "PUBLIC|PRIVATE")}
 * that omitted {@code RESTRICTED}, so a legitimate value was rejected with a 400 before
 * the request ever reached the service. These tests pin that the field is now validated
 * against the {@code ProjectVisibility} enum:
 * <ul>
 *   <li>{@code RESTRICTED} passes validation and reaches the service (the fix);</li>
 *   <li>an unknown value is rejected with a 400 whose message lists the valid tiers;</li>
 *   <li>a missing/null visibility is allowed (PATCH is partial).</li>
 * </ul>
 *
 * <p>Built with {@code standaloneSetup} rather than {@code @SpringBootTest} /
 * {@code @WebMvcTest}: it wires the real controller, the real {@code @Valid} chain (so
 * the real {@link com.federated.fl_platform_api.validation.ValueOfEnum} constraint runs)
 * and the real {@link GlobalExceptionHandler}, with no Spring context, no servlet
 * filters and no Testcontainers datasource. That keeps the test off the shared-database
 * create-drop path that the full-context integration tests share. Authorization is
 * covered by the membership/workflow integration tests, not here.
 */
class ProjectVisibilityUpdateWebTest {

    private ProjectService projectService;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        projectService = mock(ProjectService.class);
        ProjectDeletionService projectDeletionService = mock(ProjectDeletionService.class);

        ProjectController controller = new ProjectController();
        ReflectionTestUtils.setField(controller, "projectService", projectService);
        ReflectionTestUtils.setField(controller, "projectDeletionService", projectDeletionService);

        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();

        // A JavaTime-aware ObjectMapper so the GlobalExceptionHandler's ApiError
        // (which carries an Instant timestamp) serialises cleanly in this contextless setup.
        ObjectMapper objectMapper = Jackson2ObjectMapperBuilder.json().build();

        mockMvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .setMessageConverters(new MappingJackson2HttpMessageConverter(objectMapper))
                .build();
    }

    private static ProjectResponseDto dtoWithVisibility(String visibility) {
        ProjectResponseDto dto = new ProjectResponseDto();
        dto.setId(UUID.randomUUID());
        dto.setName("proj");
        dto.setVisibility(visibility);
        return dto;
    }

    @Test
    void restrictedVisibility_passesValidation_reachesService_andReturns200() throws Exception {
        UUID id = UUID.randomUUID();
        when(projectService.updateProject(eq(id), any(UpdateProjectRequest.class)))
                .thenReturn(dtoWithVisibility("RESTRICTED"));

        mockMvc.perform(patch("/api/projects/" + id)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"visibility\":\"RESTRICTED\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.visibility").value("RESTRICTED"));

        // The service is reached only if validation accepted RESTRICTED — that is the bug fix.
        ArgumentCaptor<UpdateProjectRequest> captor = ArgumentCaptor.forClass(UpdateProjectRequest.class);
        verify(projectService).updateProject(eq(id), captor.capture());
        assertEquals("RESTRICTED", captor.getValue().getVisibility());
    }

    @Test
    void unknownVisibility_isRejected_beforeService_withMessageListingEveryTier() throws Exception {
        UUID id = UUID.randomUUID();

        mockMvc.perform(patch("/api/projects/" + id)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"visibility\":\"BOGUS\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.message").value("Validation failed"))
                .andExpect(jsonPath("$.fieldErrors.visibility", containsString("PUBLIC")))
                .andExpect(jsonPath("$.fieldErrors.visibility", containsString("RESTRICTED")))
                .andExpect(jsonPath("$.fieldErrors.visibility", containsString("PRIVATE")));

        verify(projectService, never()).updateProject(any(UUID.class), any(UpdateProjectRequest.class));
    }

    @Test
    void missingVisibility_isAllowed_andReachesServiceWithNullVisibility() throws Exception {
        UUID id = UUID.randomUUID();
        when(projectService.updateProject(eq(id), any(UpdateProjectRequest.class)))
                .thenReturn(dtoWithVisibility("PUBLIC"));

        mockMvc.perform(patch("/api/projects/" + id)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"name\":\"renamed-project\"}"))
                .andExpect(status().isOk());

        ArgumentCaptor<UpdateProjectRequest> captor = ArgumentCaptor.forClass(UpdateProjectRequest.class);
        verify(projectService).updateProject(eq(id), captor.capture());
        assertEquals("renamed-project", captor.getValue().getName());
        assertNull(captor.getValue().getVisibility(),
                "a PATCH that omits visibility must leave it null (unchanged)");
    }
}
