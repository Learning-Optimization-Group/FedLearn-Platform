package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.security.InternalCallbackAuthorization;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.UUID;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Authenticated per-run model publication with immutable replay validation. */
@Service
public class RunArtifactIngestService {
    private final InternalCallbackAuthorization authorization;
    private final ArtifactRegistryService registry;
    private final ModelArtifactRepository artifacts;
    private final ObjectMapper objectMapper;

    public RunArtifactIngestService(InternalCallbackAuthorization authorization, ArtifactRegistryService registry,
                                   ModelArtifactRepository artifacts, ObjectMapper objectMapper) {
        this.authorization = authorization;
        this.registry = registry;
        this.artifacts = artifacts;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public ModelArtifact register(UUID projectId, InternalCallbackContext context, byte[] content, ArtifactKind kind,
                                  String recipeKey, String baseModelRef, String licenseTag, String evalCard) {
        var authorized = authorization.requireActiveRun(projectId, context);
        var project = authorized.project();
        var run = authorized.run();
        if (recipeKey != null && !recipeKey.equals(run.getRecipeKey())) {
            throw new IllegalArgumentException("Artifact recipe must match the recorded run");
        }
        ArtifactKind expectedKind = "LLM_LORA".equals(run.getRecipeKey())
                ? ArtifactKind.LORA_ADAPTER : ArtifactKind.FULL_CHECKPOINT;
        if (kind != expectedKind) {
            throw new IllegalArgumentException("Artifact kind must match the recorded recipe");
        }
        var existing = artifacts.findByRunIdAndKind(run.getId(), kind);
        if (existing.isPresent()) {
            ModelArtifact previous = existing.get();
            if (!Objects.equals(previous.getProjectId(), projectId)
                    || !Objects.equals(previous.getOrgId(), project.getOrgId())
                    || !Objects.equals(previous.getRecipeKey(), run.getRecipeKey())
                    || !Objects.equals(previous.getBlobSha256(), sha256(content))
                    || !Objects.equals(previous.getBaseModelRef(), baseModelRef)
                    || !Objects.equals(previous.getLicenseTag(), licenseTag)
                    || !sameEvalCard(previous.getEvalCardJson(), evalCard)) {
                throw new IllegalStateException("Artifact replay conflicts with the recorded output");
            }
        }
        // Keep the registry's validation, privacy-trace checks and lineage behavior on both paths.
        return registry.register(project.getOrgId(), projectId, run.getId(), content, kind,
                run.getRecipeKey(), baseModelRef, licenseTag, evalCard);
    }

    private boolean sameEvalCard(String first, String second) {
        if (Objects.equals(first, second)) return true;
        if (first == null || second == null) return false;
        try {
            var reader = objectMapper.reader().with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
            JsonNode firstJson = reader.readTree(first);
            JsonNode secondJson = reader.readTree(second);
            return firstJson != null && secondJson != null && firstJson.equals(secondJson);
        } catch (JsonProcessingException invalidJson) {
            return false;
        }
    }

    private static String sha256(byte[] content) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(content));
        } catch (NoSuchAlgorithmException unavailable) {
            throw new IllegalStateException("SHA-256 unavailable", unavailable);
        }
    }
}
