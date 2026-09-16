package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.model.ArtifactKind;
import com.federated.fl_platform_api.model.ModelArtifact;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.repository.ModelArtifactRepository;
import com.federated.fl_platform_api.service.ArtifactBlobStore;
import com.federated.fl_platform_api.service.RegistryModelResolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * BA-11: the shared registry→filesystem resolver used by both inference and the FL-server warm-start.
 * Materializes the FULL_CHECKPOINT head to a sha256-keyed cache (byte-exact, reused), excludes LoRA, and
 * fails LOUD on a corrupt/unreadable blob rather than masking it with the .npz fallback.
 */
class RegistryModelResolverTest {

    private final ModelArtifactRepository artifacts = mock(ModelArtifactRepository.class);
    private final ArtifactBlobStore blobStore = mock(ArtifactBlobStore.class);

    private RegistryModelResolver resolver(String cacheDir) {
        return new RegistryModelResolver(artifacts, blobStore, cacheDir);
    }

    private Project project(String modelType) {
        Project p = new Project();
        p.setId(UUID.randomUUID());
        p.setOrgId(UUID.randomUUID());
        p.setModelType(modelType);
        return p;
    }

    private ModelArtifact head(UUID projectId, String sha) {
        ModelArtifact a = new ModelArtifact();
        a.setId(UUID.randomUUID());
        a.setProjectId(projectId);
        a.setKind(ArtifactKind.FULL_CHECKPOINT);
        a.setBlobSha256(sha);
        a.setCreatedAt(Instant.now());
        return a;
    }

    @Test
    void resolveModelPath_materializesTheHeadBlobByteExact(@TempDir Path cache) throws Exception {
        Project p = project("CNN");
        String sha = "a".repeat(64);
        byte[] blob = "npz-bytes-verbatim".getBytes();
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(p.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(head(p.getId(), sha)));
        when(blobStore.get(sha)).thenReturn(blob);

        Optional<String> path = resolver(cache.toString()).resolveModelPath(p);

        assertThat(path).isPresent();
        assertThat(path.get()).endsWith(sha + ".npz");
        assertThat(Files.readAllBytes(Path.of(path.get()))).isEqualTo(blob);
    }

    @Test
    void resolveModelPath_reusesTheCacheFile_secondCallDoesNotRefetch(@TempDir Path cache) {
        Project p = project("CNN");
        String sha = "a".repeat(64);
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(p.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(head(p.getId(), sha)));
        when(blobStore.get(sha)).thenReturn("bytes".getBytes());

        RegistryModelResolver r = resolver(cache.toString());
        String first = r.resolveModelPath(p).orElseThrow();
        String second = r.resolveModelPath(p).orElseThrow();

        assertThat(second).isEqualTo(first);
        verify(blobStore, times(1)).get(sha); // content is immutable — the cache file is reused
    }

    @Test
    void resolveModelPath_isEmptyForLora_withoutTouchingTheRegistry() {
        Project p = project("LLM_LORA");
        assertThat(resolver("unused").resolveModelPath(p)).isEmpty();
        verify(artifacts, never()).findFirstByProjectIdAndKindOrderByCreatedAtDesc(any(), any());
    }

    @Test
    void resolveModelPath_isEmptyWhenNoArtifactExists() {
        Project p = project("CNN");
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(p.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.empty());
        assertThat(resolver("unused").resolveModelPath(p)).isEmpty();
    }

    @Test
    void resolveModelPath_failsLoudOnACorruptOrUnreadableBlob(@TempDir Path cache) {
        Project p = project("CNN");
        String sha = "a".repeat(64);
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(p.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(head(p.getId(), sha)));
        // Chunk B integrity-on-read throws unchecked when the on-disk bytes don't hash to the key.
        when(blobStore.get(sha)).thenThrow(new IllegalStateException("blob integrity check failed"));

        assertThatThrownBy(() -> resolver(cache.toString()).resolveModelPath(p))
                .isInstanceOf(IllegalStateException.class); // NOT silently masked by the .npz fallback
    }

    @Test
    void hasModel_reflectsTheRegistryHead_andIsFalseForLora() {
        Project cnn = project("CNN");
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(cnn.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.of(head(cnn.getId(), "a".repeat(64))));
        assertThat(resolver("unused").hasModel(cnn)).isTrue();

        Project none = project("CNN");
        when(artifacts.findFirstByProjectIdAndKindOrderByCreatedAtDesc(none.getId(), ArtifactKind.FULL_CHECKPOINT))
                .thenReturn(Optional.empty());
        assertThat(resolver("unused").hasModel(none)).isFalse();

        assertThat(resolver("unused").hasModel(project("LLM_LORA"))).isFalse();
    }
}
