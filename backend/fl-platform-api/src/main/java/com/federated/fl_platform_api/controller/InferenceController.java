package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.InferableModelDto;
import com.federated.fl_platform_api.dto.InferenceRequest;
import com.federated.fl_platform_api.dto.InferenceResultDto;
import com.federated.fl_platform_api.service.InferenceService;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

/**
 * "Use a model" endpoints — run inference against a trained federated model.
 * Authenticated by default (not in {@code SecurityConfig} public paths); per-project
 * access is enforced in {@link InferenceService}/{@code ProjectService}.
 */
@RestController
@RequestMapping("/api/inference")
public class InferenceController {

    private final InferenceService inferenceService;

    public InferenceController(InferenceService inferenceService) {
        this.inferenceService = inferenceService;
    }

    /** Trained models the current user can run, with their input kind + class labels. */
    @GetMapping("/models")
    public ResponseEntity<List<InferableModelDto>> listModels() {
        return ResponseEntity.ok(inferenceService.listInferableModels());
    }

    /** Run one inference on the given project's model. */
    @PostMapping("/{projectId}")
    public ResponseEntity<InferenceResultDto> infer(@PathVariable @NonNull UUID projectId,
                                                    @Valid @RequestBody InferenceRequest request) {
        return ResponseEntity.ok(inferenceService.runInference(projectId, request));
    }
}
