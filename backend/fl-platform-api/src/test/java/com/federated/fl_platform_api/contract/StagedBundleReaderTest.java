package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * A staged bundle's contract facts are read from its manifest and re-checked against the files on disk, so a
 * contract never binds a program that is not there exactly as described.
 */
class StagedBundleReaderTest {

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final UUID RUN = UUID.randomUUID();

    private static StagedBundleReader reader(Path root) {
        StagedBundleReader reader = new StagedBundleReader();
        ReflectionTestUtils.setField(reader, "modelBundleDir", root.toString());
        return reader;
    }

    /** Writes three programs and a manifest describing them; returns the manifest for further edits. */
    private static ObjectNode stage(Path root) throws Exception {
        Path dir = Files.createDirectories(root.resolve(RUN.toString()));
        ObjectNode manifest = JSON.createObjectNode();
        ArrayNode files = manifest.putArray("modelFiles");
        for (String name : List.of("loss.pte", "infer.pte", "trainable.pte")) {
            byte[] bytes = ("program " + name).getBytes();
            Files.write(dir.resolve(name), bytes);
            files.addObject().put("file", name).put("sha256", ExecutionContractStore.sha256(bytes))
                    .put("byteSize", bytes.length);
        }
        manifest.putArray("requiredOperators").add("aten::addmm.out").add("aten::relu.out");
        manifest.putObject("resourceEnvelope").put("peakMemoryBytes", 67108864).put("storageBytes", 100)
                .put("probeMs", 2000).put("trainMs", 10000).put("basis", "budget");
        ArrayNode layout = manifest.putObject("modelManifest").putArray("paramLayout");
        layout.addObject().put("name", "fc1.weight").putArray("shape").add(5).add(4);
        layout.addObject().put("name", "fc1.bias").putArray("shape").add(5);
        manifest.put("maxBatch", 8);
        write(root, manifest);
        return manifest;
    }

    private static void write(Path root, ObjectNode manifest) throws IOException {
        Files.writeString(root.resolve(RUN.toString()).resolve("manifest.json"), JSON.writeValueAsString(manifest));
    }

    @Test
    void theManifestsContractFactsAreRead(@TempDir Path root) throws Exception {
        stage(root);
        StagedBundle bundle = reader(root).read(RUN);
        assertThat(bundle.modelFiles()).extracting(StagedBundle.ModelFile::file)
                .containsExactly("loss.pte", "infer.pte", "trainable.pte");
        assertThat(bundle.requiredOperators()).containsExactly("aten::addmm.out", "aten::relu.out");
        assertThat(bundle.envelope()).isEqualTo(new StagedBundle.ResourceEnvelope(67108864L, 100L, 2000L, 10000L));
        assertThat(bundle.paramLayout()).containsExactly(
                new StagedBundle.LayoutEntry("fc1.weight", List.of(5L, 4L)),
                new StagedBundle.LayoutEntry("fc1.bias", List.of(5L)));
        assertThat(bundle.maxBatch()).isEqualTo(8);
    }

    // The programs take 1..maxBatch examples per call; a bundle that does not say how many cannot be checked against
    // the contract's batch size, and a device would find out only when the runtime refused its data.
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"absent", "0", "-1", "2.5", "\"8\""})
    void aBundleThatDoesNotStateAPositiveMaxBatchIsRefused(String value, @TempDir Path root) throws Exception {
        ObjectNode manifest = stage(root);
        if (value.equals("absent")) {
            manifest.remove("maxBatch");
        } else {
            manifest.set("maxBatch", JSON.readTree(value));
        }
        write(root, manifest);
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class)
                .hasMessageContaining("maxBatch");
    }

    @Test
    void aProgramThatChangedOnDiskIsRefused(@TempDir Path root) throws Exception {
        stage(root);
        Files.writeString(root.resolve(RUN.toString()).resolve("trainable.pte"), "tampered");
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class)
                .hasMessageContaining("trainable.pte");
    }

    @Test
    void aMissingProgramIsRefused(@TempDir Path root) throws Exception {
        stage(root);
        Files.delete(root.resolve(RUN.toString()).resolve("infer.pte"));
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class)
                .hasMessageContaining("infer.pte");
    }

    @Test
    void aFileOutsideTheProgramAllowlistIsRefused(@TempDir Path root) throws Exception {
        ObjectNode manifest = stage(root);
        ((ArrayNode) manifest.get("modelFiles")).addObject().put("file", "../etc/passwd")
                .put("sha256", "a".repeat(64)).put("byteSize", 1);
        write(root, manifest);
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class)
                .hasMessageContaining("../etc/passwd");
    }

    @Test
    void aBundleWithoutContractFactsIsRefused(@TempDir Path root) throws Exception {
        ObjectNode manifest = stage(root);
        manifest.remove("requiredOperators");
        write(root, manifest);
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class)
                .hasMessageContaining("requiredOperators");
    }

    @Test
    void anUnstagedRunIsRefused(@TempDir Path root) {
        assertThatThrownBy(() -> reader(root).read(RUN)).isInstanceOf(IOException.class);
    }
}
