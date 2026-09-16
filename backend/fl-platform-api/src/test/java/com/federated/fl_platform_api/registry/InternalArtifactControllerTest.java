package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.controller.InternalArtifactController;
import com.federated.fl_platform_api.exception.GlobalExceptionHandler;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.service.ArtifactRegistryService;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.UUID;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Standalone MockMvc (no Spring context, no database): verifies the controller parses the multipart
 * model upload and delegates to the registry. The {@code X-Internal-Key} gate is a cross-cutting
 * filter over all {@code /api/internal/**} (wired in SecurityConfig), exercised by the existing
 * internal endpoints — it is not this controller's own logic, so it is intentionally out of scope
 * here. Using standalone setup keeps this test off the shared Testcontainers context.
 */
class InternalArtifactControllerTest {

    private final ArtifactRegistryService registry = mock(ArtifactRegistryService.class);
    private final MockMvc mvc =
            MockMvcBuilders.standaloneSetup(new InternalArtifactController(registry)).build();

    @Test
    void posts_model_bytes_and_delegates_to_the_registry() throws Exception {
        UUID projectId = UUID.randomUUID();
        ModelArtifact stub = new ModelArtifact();
        stub.setId(UUID.randomUUID());
        stub.setBlobSha256("d".repeat(64));
        when(registry.registerForProject(eq(projectId), any(byte[].class), eq(ArtifactKind.LORA_ADAPTER),
                any(), any(), any(), any())).thenReturn(stub);

        MockMultipartFile model = new MockMultipartFile(
                "model", "model.npz", "application/octet-stream", "the-model-bytes".getBytes());

        mvc.perform(multipart("/api/internal/projects/{id}/artifacts", projectId)
                        .file(model)
                        .param("kind", "LORA_ADAPTER")
                        .param("recipeKey", "LLM_LORA"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sha256").value("d".repeat(64)));

        verify(registry).registerForProject(eq(projectId), any(byte[].class), eq(ArtifactKind.LORA_ADAPTER),
                eq("LLM_LORA"), isNull(), isNull(), isNull());
    }

    /**
     * SE-11: a DP-policy refusal from the registry ({@code IllegalArgumentException}) must surface
     * to the uploading FL server as a 400 with the policy message — wired through
     * {@link GlobalExceptionHandler} exactly as in the deployed app.
     */
    @Test
    void dp_policy_rejection_surfaces_as_a_400_with_the_policy_message() throws Exception {
        UUID projectId = UUID.randomUUID();
        when(registry.registerForProject(eq(projectId), any(byte[].class), eq(ArtifactKind.LORA_ADAPTER),
                any(), any(), any(), any()))
                .thenThrow(new IllegalArgumentException(
                        "an artifact may not claim DP without a committed accountant trace"));

        MockMvc mvcWithAdvice = MockMvcBuilders
                .standaloneSetup(new InternalArtifactController(registry))
                .setControllerAdvice(new GlobalExceptionHandler())
                .build();

        MockMultipartFile model = new MockMultipartFile(
                "model", "model.npz", "application/octet-stream", "the-model-bytes".getBytes());

        mvcWithAdvice.perform(multipart("/api/internal/projects/{id}/artifacts", projectId)
                        .file(model)
                        .param("kind", "LORA_ADAPTER")
                        .param("evalCard", "{\"dp\":{\"enabled\":true}}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.message",
                        containsString("committed accountant trace")));
    }
}
