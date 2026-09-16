package com.federated.fl_platform_api.registry;

import com.federated.fl_platform_api.service.LocalFsArtifactBlobStore;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LocalFsArtifactBlobStoreTest {

    private LocalFsArtifactBlobStore store(Path root) {
        return new LocalFsArtifactBlobStore(root.toString());
    }

    @Test
    void put_returns_the_content_sha256_and_get_round_trips(@TempDir Path root) {
        LocalFsArtifactBlobStore s = store(root);
        byte[] content = "hello".getBytes(StandardCharsets.UTF_8);

        String key = s.put(content);

        // sha256("hello") — proves the store keys on the CONTENT's hash, not anything caller-supplied.
        assertThat(key).isEqualTo("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
        assertThat(s.exists(key)).isTrue();
        assertThat(s.get(key)).isEqualTo(content);
    }

    @Test
    void put_is_idempotent_write_once(@TempDir Path root) {
        LocalFsArtifactBlobStore s = store(root);
        byte[] content = "some model bytes".getBytes(StandardCharsets.UTF_8);

        String k1 = s.put(content);
        String k2 = s.put(content); // identical bytes: no exception, same key — this IS dedup

        assertThat(k2).isEqualTo(k1);
        assertThat(s.get(k1)).isEqualTo(content);
    }

    @Test
    void different_content_gets_different_keys(@TempDir Path root) {
        LocalFsArtifactBlobStore s = store(root);
        assertThat(s.put("a".getBytes(StandardCharsets.UTF_8)))
                .isNotEqualTo(s.put("b".getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void get_unknown_key_throws(@TempDir Path root) {
        LocalFsArtifactBlobStore s = store(root);
        assertThatThrownBy(() -> s.get("0".repeat(64))).isInstanceOf(RuntimeException.class);
    }

    @Test
    void malformed_key_is_rejected(@TempDir Path root) {
        LocalFsArtifactBlobStore s = store(root);
        assertThatThrownBy(() -> s.exists("not-a-hash")).isInstanceOf(IllegalArgumentException.class);
    }

    // BA-11: content-addressing integrity on READ — a corrupted/tampered blob must never be served as
    // the right bytes under the right id.
    @Test
    void get_fails_loud_when_the_stored_blob_is_corrupted_on_disk(@TempDir Path root) throws Exception {
        LocalFsArtifactBlobStore s = store(root);
        byte[] content = "trusted model bytes".getBytes(StandardCharsets.UTF_8);
        String key = s.put(content);

        // Overwrite the on-disk blob at its content-addressed path (bit-rot / swapped file).
        Path blobPath = root.resolve(key.substring(0, 2)).resolve(key.substring(2, 4)).resolve(key);
        java.nio.file.Files.write(blobPath, "TAMPERED".getBytes(StandardCharsets.UTF_8));

        assertThatThrownBy(() -> s.get(key))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("integrity check failed");
    }
}
