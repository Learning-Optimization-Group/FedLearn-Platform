package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.UUID;

/**
 * Reads the contract facts of a run's staged bundle ({@code <app.model-bundle.dir>/<runId>/manifest.json}, written
 * by scripts/stage_model_bundle.py) and re-checks every model program on disk against the digest and size the
 * manifest records, so a contract never binds a program that is not there exactly as described.
 */
@Component
public class StagedBundleReader {

    /** The only files a bundle's contract facts may name: the served model programs. */
    static final Set<String> MODEL_PROGRAMS = Set.of("loss.pte", "infer.pte", "trainable.pte");

    private static final ObjectMapper JSON = new ObjectMapper();

    @Value("${app.model-bundle.dir:/var/models}")
    private String modelBundleDir;

    public StagedBundle read(UUID runId) throws IOException {
        Path dir = Path.of(modelBundleDir, runId.toString());
        JsonNode manifest = JSON.readTree(dir.resolve("manifest.json").toFile());
        List<StagedBundle.ModelFile> files = new ArrayList<>();
        for (JsonNode f : required(manifest, "modelFiles")) {
            String name = f.path("file").asText();
            if (!MODEL_PROGRAMS.contains(name)) {
                throw new IOException("the staged bundle names an unexpected model file " + name);
            }
            StagedBundle.ModelFile file = new StagedBundle.ModelFile(name, f.path("sha256").asText(),
                    f.path("byteSize").asLong());
            verify(dir.resolve(name), file);
            files.add(file);
        }
        List<String> operators = new ArrayList<>();
        for (JsonNode op : required(manifest, "requiredOperators")) {
            operators.add(op.asText());
        }
        JsonNode e = required(manifest, "resourceEnvelope");
        StagedBundle.ResourceEnvelope envelope = new StagedBundle.ResourceEnvelope(
                e.path("peakMemoryBytes").asLong(), e.path("storageBytes").asLong(),
                e.path("probeMs").asLong(), e.path("trainMs").asLong());
        List<StagedBundle.LayoutEntry> layout = new ArrayList<>();
        for (JsonNode entry : required(manifest.path("modelManifest"), "paramLayout")) {
            List<Long> shape = new ArrayList<>();
            entry.path("shape").forEach(extent -> shape.add(extent.asLong()));
            layout.add(new StagedBundle.LayoutEntry(entry.path("name").asText(), shape));
        }
        return new StagedBundle(files, operators, envelope, layout);
    }

    private static JsonNode required(JsonNode node, String field) throws IOException {
        JsonNode value = node.get(field);
        if (value == null || value.isNull() || (value.isContainerNode() && value.isEmpty())) {
            throw new IOException("the staged bundle carries no " + field);
        }
        return value;
    }

    private static void verify(Path path, StagedBundle.ModelFile expected) throws IOException {
        if (!Files.isRegularFile(path)) {
            throw new IOException("the staged bundle has no " + expected.file() + " on disk");
        }
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
        long size = 0;
        try (InputStream in = new DigestInputStream(Files.newInputStream(path), digest)) {
            byte[] buffer = new byte[1 << 16];
            for (int n; (n = in.read(buffer)) != -1; ) {
                size += n;
            }
        }
        String actual = HexFormat.of().formatHex(digest.digest());
        if (!actual.equals(expected.sha256()) || size != expected.byteSize()) {
            throw new IOException("the staged " + expected.file() + " no longer matches its recorded digest and size");
        }
    }
}
