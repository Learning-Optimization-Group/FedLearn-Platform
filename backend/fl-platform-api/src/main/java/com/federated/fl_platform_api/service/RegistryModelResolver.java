package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Optional;

/**
 * BA-11: resolves a project's current model from the content-addressed registry (the source of truth)
 * to a local filesystem path, so both the inference path ({@code ProjectService.resolveInferenceTarget})
 * and the FL-server warm-start ({@code FlowerServerManager}) read the registry instead of the
 * overwritable {@code .npz}. Extracted as a shared bean because {@code FlowerServerManager} cannot depend
 * on {@code ProjectService} (that would be a circular reference, which Spring Boot rejects by default).
 *
 * <p>Scoped to {@link ArtifactKind#FULL_CHECKPOINT}: a LoRA project's head is a safetensors bundle that
 * the numpy-{@code .npz} read paths cannot parse, so LoRA keeps its {@code .npz} (this returns empty).
 * The caller is responsible for org-scope authorization on the {@link Project} — an artifact's org always
 * equals its project's org (set at registration), so no separate artifact-org check is needed here.</p>
 */
@Service
public class RegistryModelResolver {

    private static final Logger log = LoggerFactory.getLogger(RegistryModelResolver.class);

    private final ModelArtifactRepository artifacts;
    private final ArtifactBlobStore blobStore;
    private final String cacheDir;

    public RegistryModelResolver(ModelArtifactRepository artifacts, ArtifactBlobStore blobStore,
                                 @Value("${app.model-blob-cache.dir:models/blob-cache}") String cacheDir) {
        this.artifacts = artifacts;
        this.blobStore = blobStore;
        this.cacheDir = cacheDir;
    }

    /** True if the registry holds a usable FULL_CHECKPOINT head for this project (existence-only, no copy). */
    public boolean hasModel(Project project) {
        return headArtifact(project).isPresent();
    }

    /**
     * Resolve the project's registry head to a local filesystem path (materialized, integrity-checked),
     * or empty to fall back to the {@code .npz}. Empty for LoRA and for a project with no artifact yet
     * (the common migration case). A local cache-WRITE failure also degrades to empty. But a blob-store
     * read failure or an integrity mismatch ({@link ArtifactBlobStore#get} throws unchecked) is allowed to
     * PROPAGATE — a corrupt or unreadable registry head must fail loud, never be silently masked by the
     * {@code .npz} fallback (that fallback means "no artifact", not "artifact unreadable").
     */
    public Optional<String> resolveModelPath(Project project) {
        ModelArtifact head = headArtifact(project).orElse(null);
        if (head == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(materializeBlob(head.getBlobSha256()));
        } catch (IOException e) {
            log.warn("BA-11: could not materialize registry blob {} for project {}; falling back to .npz",
                    head.getBlobSha256(), project.getId(), e);
            return Optional.empty();
        }
    }

    private Optional<ModelArtifact> headArtifact(Project project) {
        if ("LLM_LORA".equals(project.getModelType())) {
            return Optional.empty(); // safetensors head — the .npz read paths handle LoRA
        }
        return artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(
                project.getId(), ArtifactKind.FULL_CHECKPOINT);
    }

    /**
     * Materialize a content-addressed blob to a stable, sha256-named cache file and return its absolute
     * path. Idempotent: the content is immutable, so an existing cache file is reused and concurrent
     * writers converge on identical bytes (a lost publish race just keeps the winner's file).
     */
    private String materializeBlob(String sha256) throws IOException {
        File dir = new File(cacheDir);
        File cached = new File(dir, sha256 + ".npz");
        if (cached.isFile()) {
            return cached.getAbsolutePath();
        }
        Files.createDirectories(dir.toPath());
        byte[] bytes = blobStore.get(sha256); // integrity-checked on read (BA-11 Chunk B)
        File tmp = File.createTempFile(sha256 + "-", ".npz.part", dir);
        try {
            Files.write(tmp.toPath(), bytes);
            Files.move(tmp.toPath(), cached.toPath(), StandardCopyOption.ATOMIC_MOVE);
        } catch (java.nio.file.FileAlreadyExistsException race) {
            // another request published the identical-content file first — fine.
        } finally {
            Files.deleteIfExists(tmp.toPath());
        }
        return cached.getAbsolutePath();
    }
}
