package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import com.federated.fl_platform_api.service.RunArtifactIngestService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestAttribute;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;
import java.util.UUID;

/**
 * Internal callback that {@code fl_server.py} posts a run's final model to when it finishes. Gated by
 * {@code InternalApiKeyFilter}; verified run authorization is rechecked inside the write transaction.
 * Registers a versioned, content-addressed artifact — write-new-not-overwrite (DA-2).
 */
@RestController
@RequestMapping("/api/internal/projects")
public class InternalArtifactController {

    private final RunArtifactIngestService ingest;

    public InternalArtifactController(RunArtifactIngestService ingest) {
        this.ingest = ingest;
    }

    @PostMapping(path = "/{projectId}/artifacts", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, String>> registerArtifact(
            @PathVariable UUID projectId,
            @RequestAttribute(InternalCallbackContext.REQUEST_ATTRIBUTE) InternalCallbackContext context,
            @RequestParam("model") MultipartFile model,
            @RequestParam("kind") ArtifactKind kind,
            @RequestParam(value = "recipeKey", required = false) String recipeKey,
            @RequestParam(value = "baseModelRef", required = false) String baseModelRef,
            @RequestParam(value = "licenseTag", required = false) String licenseTag,
            @RequestParam(value = "evalCard", required = false) String evalCard) throws IOException {

        ModelArtifact artifact = ingest.register(
                projectId, context, model.getBytes(), kind, recipeKey, baseModelRef, licenseTag, evalCard);

        return ResponseEntity.ok(Map.of(
                "id", artifact.getId().toString(),
                "sha256", artifact.getBlobSha256()));
    }
}
