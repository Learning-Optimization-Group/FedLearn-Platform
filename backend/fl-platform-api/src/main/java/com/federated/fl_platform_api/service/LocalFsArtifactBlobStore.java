package com.federated.fl_platform_api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Local-filesystem {@link ArtifactBlobStore}. Blobs live under a two-level sha256 fan-out
 * ({@code root/ab/cd/abcd...}) and are written atomically (temp file + atomic rename) so a crash
 * can never leave a partial blob at a content-addressed path. Writes are content-verified by
 * construction (the key is the hash of the bytes) and idempotent (an existing key is a no-op).
 */
@Service
public class LocalFsArtifactBlobStore implements ArtifactBlobStore {

    private final Path root;

    public LocalFsArtifactBlobStore(
            @Value("${app.artifact-store.root:artifact-store}") String root) {
        this.root = Path.of(root);
    }

    @Override
    public String backendId() {
        return "LOCAL_FS";
    }

    @Override
    public String put(byte[] content) {
        String key = sha256Hex(content);
        Path target = pathFor(key);
        if (Files.exists(target)) {
            return key; // idempotent write-once: identical content already stored
        }
        try {
            Files.createDirectories(target.getParent());
            Path tmp = Files.createTempFile(target.getParent(), ".tmp-", ".blob");
            try {
                Files.write(tmp, content);
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (FileAlreadyExistsException raceLost) {
                Files.deleteIfExists(tmp); // a concurrent writer won; identical bytes are already there
            } catch (IOException e) {
                Files.deleteIfExists(tmp);
                throw e;
            }
            return key;
        } catch (IOException e) {
            throw new UncheckedIOException("blob put failed for " + key, e);
        }
    }

    @Override
    public byte[] get(String sha256) {
        try {
            return Files.readAllBytes(pathFor(normalize(sha256)));
        } catch (IOException e) {
            throw new UncheckedIOException("blob get failed for " + sha256, e);
        }
    }

    @Override
    public boolean exists(String sha256) {
        return Files.exists(pathFor(normalize(sha256)));
    }

    private Path pathFor(String key) {
        // key is validated 64-hex, so the two 2-char segments are safe (no path traversal).
        return root.resolve(key.substring(0, 2)).resolve(key.substring(2, 4)).resolve(key);
    }

    private static String normalize(String hex) {
        String h = hex == null ? "" : hex.toLowerCase();
        if (!h.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("not a sha256 hex key: " + hex);
        }
        return h;
    }

    static String sha256Hex(byte[] content) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(content);
            StringBuilder sb = new StringBuilder(64);
            for (byte b : digest) {
                sb.append(Character.forDigit((b >> 4) & 0xF, 16));
                sb.append(Character.forDigit(b & 0xF, 16));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }
}
