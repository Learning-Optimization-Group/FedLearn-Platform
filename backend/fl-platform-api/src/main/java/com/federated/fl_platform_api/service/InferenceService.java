package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.GenerationRequest;
import com.federated.fl_platform_api.dto.GenerationResultDto;
import com.federated.fl_platform_api.dto.InferableModelDto;
import com.federated.fl_platform_api.dto.InferenceRequest;
import com.federated.fl_platform_api.dto.InferenceResultDto;
import com.federated.fl_platform_api.exception.InferenceBusyException;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ServerProcessException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Runs single-input inference on a trained model by spawning the {@code infer.py}
 * wrapper (mirrors {@link ModelInitializer}'s process model). Stateless — no DB
 * writes. The Python side writes its JSON result to an out-file (not stdout) so
 * torch/wrapper log noise can't corrupt the contract.
 */
@Service
public class InferenceService {

    private static final Logger log = LoggerFactory.getLogger(InferenceService.class);

    /** Generous cap: torch cold-start dominates; a single forward pass is sub-second. */
    private static final long PROCESS_TIMEOUT_SECONDS = 120;
    /** ~9 MB of decoded image bytes. Guards against oversized uploads. */
    private static final int MAX_IMAGE_BYTES = 9 * 1024 * 1024;
    /** Sanity bound on feature-vector length (real models use ≤ a few thousand). */
    private static final int MAX_VECTOR_LENGTH = 100_000;

    private final ProjectService projectService;
    private final WebSocketService webSocketService;
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * Bounds how many torch inference processes can run at once. Each process is
     * heavy (hundreds of MB + a full forward pass), so an uncapped fan-out of
     * authenticated requests would fork-bomb the host into resource exhaustion.
     * Excess callers get a fast 503 instead of piling on.
     */
    private final Semaphore inferenceSlots;
    private final long generationTimeoutSeconds;

    /** Live generation subprocesses, keyed by projectId — enables Stop. Mirrors FlowerServerManager.runningServers. */
    final Map<UUID, Process> runningGenerations = new ConcurrentHashMap<>();
    /** Projects whose in-flight generation was user-stopped (so runGenerationScript returns a "stopped" result, not a 502). */
    final Set<UUID> stoppedGenerations = ConcurrentHashMap.newKeySet();

    @Value("${python.script.infer.path:src/main/resources/scripts/run_infer.sh}")
    private String inferWrapperPath;

    public InferenceService(ProjectService projectService,
                            WebSocketService webSocketService,
                            @Value("${inference.max-concurrent:2}") int maxConcurrent,
                            @Value("${inference.generation-timeout-seconds:300}") long generationTimeoutSeconds) {
        this.projectService = projectService;
        this.webSocketService = webSocketService;
        this.inferenceSlots = new Semaphore(Math.max(1, maxConcurrent), true);
        this.generationTimeoutSeconds = generationTimeoutSeconds;
    }

    public List<InferableModelDto> listInferableModels() {
        return projectService.listInferableModels();
    }

    /**
     * Resolve (with authz), validate input, run the model, return the result.
     *
     * @throws com.federated.fl_platform_api.exception.ResourceNotFoundException project not visible (404)
     * @throws org.springframework.security.access.AccessDeniedException caller not a participant (403)
     * @throws ProjectStateException model not trained / type unsupported (409)
     * @throws IllegalArgumentException malformed input (400)
     * @throws ServerProcessException python process failed/timed out (502)
     */
    public InferenceResultDto runInference(@NonNull UUID projectId, InferenceRequest request) {
        ProjectService.InferenceTarget target = projectService.resolveInferenceTarget(projectId);

        String inputKind = projectService.inputKindFor(target.modelType());
        if (!isInteractiveInputKind(inputKind)) {
            throw new ProjectStateException(
                    "Interactive inference is not supported for model type '" + target.modelType() + "' yet.");
        }

        Path inputFile = null;
        Path imageFile = null;
        Path outputFile = null;
        try {
            outputFile = Files.createTempFile("fedlearn-infer-out", ".json");

            Map<String, Object> payload;
            if ("image".equals(inputKind)) {
                byte[] bytes = decodeImage(request);
                imageFile = Files.createTempFile("fedlearn-infer-img", ".img");
                Files.write(imageFile, bytes);
                payload = Map.of("kind", "image", "imagePath", imageFile.toAbsolutePath().toString());
            } else if ("vector".equals(inputKind)) {
                List<Double> values = validateVector(request);
                payload = Map.of("kind", "vector", "values", values);
            } else { // text
                payload = Map.of("kind", "text", "text", validateText(request));
            }

            inputFile = Files.createTempFile("fedlearn-infer-in", ".json");
            objectMapper.writeValue(inputFile.toFile(), payload);

            // Bound concurrent torch processes — fast-fail with 503 when saturated.
            if (!inferenceSlots.tryAcquire()) {
                throw new InferenceBusyException(
                        "Inference is at capacity right now. Please retry in a few seconds.");
            }
            JsonNode result;
            try {
                result = runScript(target, inputFile, outputFile);
            } finally {
                inferenceSlots.release();
            }
            return toDto(result, target.modelType());

        } catch (IOException e) {
            throw new ServerProcessException("Inference I/O failure", e);
        } finally {
            deleteQuietly(inputFile);
            deleteQuietly(imageFile);
            deleteQuietly(outputFile);
        }
    }

    // ─── input validation ────────────────────────────────────────────────────

    /**
     * Returns {@code true} for input kinds that are fully wired through {@code infer.py}
     * and therefore eligible for interactive inference: image, vector, and text.
     */
    static boolean isInteractiveInputKind(String inputKind) {
        return "image".equals(inputKind) || "vector".equals(inputKind) || "text".equals(inputKind);
    }

    private byte[] decodeImage(InferenceRequest request) {
        String b64 = request.getImageBase64();
        if (b64 == null || b64.isBlank()) {
            throw new IllegalArgumentException("imageBase64 is required for this model.");
        }
        // Strip an optional data-URL prefix (e.g. "data:image/png;base64,").
        int comma = b64.indexOf(',');
        if (b64.startsWith("data:") && comma >= 0) {
            b64 = b64.substring(comma + 1);
        }
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(b64.strip());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("imageBase64 is not valid base64.");
        }
        if (bytes.length == 0) {
            throw new IllegalArgumentException("Decoded image is empty.");
        }
        if (bytes.length > MAX_IMAGE_BYTES) {
            throw new IllegalArgumentException("Image too large (max " + (MAX_IMAGE_BYTES / (1024 * 1024)) + " MB).");
        }
        return bytes;
    }

    private List<Double> validateVector(InferenceRequest request) {
        List<Double> values = request.getValues();
        if (values == null || values.isEmpty()) {
            throw new IllegalArgumentException("A numeric 'values' vector is required for this model.");
        }
        if (values.size() > MAX_VECTOR_LENGTH) {
            throw new IllegalArgumentException("Vector too long (max " + MAX_VECTOR_LENGTH + " values).");
        }
        for (Double v : values) {
            if (v == null || v.isNaN() || v.isInfinite()) {
                throw new IllegalArgumentException("Vector contains a non-finite value.");
            }
        }
        return values;
    }

    private String validateText(InferenceRequest request) {
        String text = request.getText();
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("A non-empty 'text' string is required for this model.");
        }
        return text;
    }

    // ─── process execution ───────────────────────────────────────────────────

    private JsonNode runScript(ProjectService.InferenceTarget target, Path inputFile, Path outputFile) {
        File wrapper = new File(inferWrapperPath);
        String absWrapper = wrapper.getAbsolutePath();

        List<String> command = new ArrayList<>();
        if (!System.getProperty("os.name").toLowerCase().contains("win")) {
            command.add("bash");
        }
        command.add(absWrapper);
        command.add("--model-path"); command.add(target.modelPath());
        command.add("--model-type"); command.add(target.modelType());
        command.add("--model-name"); command.add(target.modelName() == null ? "" : target.modelName());
        command.add("--in"); command.add(inputFile.toAbsolutePath().toString());
        command.add("--out"); command.add(outputFile.toAbsolutePath().toString());

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        pb.redirectErrorStream(true);

        StringBuilder diag = new StringBuilder();
        Process process;
        try {
            log.debug("Spawning inference for {}/{}", target.modelType(), target.modelName());
            process = pb.start();
        } catch (IOException e) {
            throw new ServerProcessException("Failed to start inference process", e);
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                diag.append(line).append('\n');
            }
            boolean finished = process.waitFor(PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new ServerProcessException("Inference timed out after " + PROCESS_TIMEOUT_SECONDS + "s");
            }
        } catch (IOException e) {
            throw new ServerProcessException("Inference process I/O error", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
            throw new ServerProcessException("Inference interrupted", e);
        }

        // Parse the result file regardless of exit code (the script writes ok=false on handled errors).
        JsonNode result = readResult(outputFile);
        if (result == null) {
            log.warn("Inference produced no result file. Output:\n{}", diag);
            throw new ServerProcessException("Inference produced no result (exit=" + process.exitValue() + ")");
        }
        if (!result.path("ok").asBoolean(false)) {
            String err = result.path("error").asText("Inference failed");
            // The script tags each failure: "input" = bad caller payload (safe to
            // surface as 400), anything else = a server-side fault (model load,
            // arch import, torch error) that must NOT leak internals to the client.
            String errorKind = result.path("errorKind").asText("internal");
            if ("input".equals(errorKind)) {
                log.info("Inference rejected bad input for {}: {}", target.modelType(), err);
                throw new IllegalArgumentException(err);
            }
            // Server-side: log the real reason, return a generic 502 (no path/stack leak).
            log.warn("Inference internal failure for {}: {}", target.modelType(), err);
            throw new ServerProcessException("Inference failed while executing the model");
        }
        return result;
    }

    private JsonNode readResult(Path outputFile) {
        try {
            if (outputFile == null || !Files.exists(outputFile) || Files.size(outputFile) == 0) {
                return null;
            }
            return objectMapper.readTree(outputFile.toFile());
        } catch (IOException e) {
            log.warn("Failed to read inference result file: {}", e.getMessage());
            return null;
        }
    }

    private InferenceResultDto toDto(JsonNode r, String modelType) {
        InferenceResultDto dto = new InferenceResultDto();
        dto.setModelType(r.path("modelType").asText(modelType));
        dto.setPredictedIndex(r.path("predictedIndex").asInt());
        dto.setPredictedLabel(r.path("predictedLabel").asText());
        dto.setClasses(toStringList(r.path("classes")));
        dto.setProbabilities(toDoubleList(r.path("probabilities")));
        dto.setLogits(toDoubleList(r.path("logits")));
        return dto;
    }

    private List<String> toStringList(JsonNode arr) {
        List<String> out = new ArrayList<>();
        if (arr != null && arr.isArray()) {
            arr.forEach(n -> out.add(n.asText()));
        }
        return out;
    }

    private List<Double> toDoubleList(JsonNode arr) {
        List<Double> out = new ArrayList<>();
        if (arr != null && arr.isArray()) {
            arr.forEach(n -> out.add(n.asDouble()));
        }
        return out;
    }

    // ─── streaming text generation ───────────────────────────────────────────

    /**
     * If the stdout line is a {"token":…} JSON object, rebroadcast it to the
     * inference topic. Package-private for unit testability.
     */
    boolean broadcastIfToken(UUID projectId, String line) {
        try {
            JsonNode n = objectMapper.readTree(line);
            if (n.isObject() && n.has("token")) {
                webSocketService.sendInferenceToken(projectId, line);
                return true;
            }
        } catch (IOException ignored) {
            // non-JSON diagnostic line — not a token
        }
        return false;
    }

    /**
     * Spawn {@code infer.py} in generation mode, stream each token chunk to
     * {@code /topic/inference/{projectId}}, and return the final result.
     *
     * @throws ProjectStateException  project model type does not support generation (409)
     * @throws IllegalArgumentException blank prompt (400)
     * @throws InferenceBusyException semaphore saturated (503)
     * @throws ServerProcessException python process failed or timed out (502)
     */
    public GenerationResultDto generate(@NonNull UUID projectId, GenerationRequest request) {
        ProjectService.InferenceTarget target = projectService.resolveInferenceTarget(projectId);
        String inputKind = projectService.inputKindFor(target.modelType(), target.taskType());
        if (!"generation".equals(inputKind)) {
            throw new ProjectStateException(
                    "Text generation is not supported for this project (requires an LLM_LORA / CAUSAL_LM model).");
        }
        if (request.getPrompt() == null || request.getPrompt().isBlank()) {
            throw new IllegalArgumentException("A non-empty prompt is required.");
        }
        Path inputFile = null, outputFile = null;
        try {
            outputFile = Files.createTempFile("fedlearn-gen-out", ".json");
            inputFile = Files.createTempFile("fedlearn-gen-in", ".json");
            objectMapper.writeValue(inputFile.toFile(),
                    java.util.Map.of("kind", "generation", "prompt", request.getPrompt(),
                            "history", request.getHistory() == null ? java.util.List.of() : request.getHistory()));
            if (!inferenceSlots.tryAcquire()) {
                throw new InferenceBusyException("Inference is at capacity right now. Please retry in a few seconds.");
            }
            JsonNode result;
            try {
                result = runGenerationScript(target, inputFile, outputFile, projectId,
                        request.getMaxNewTokens(), request.getTemperature());
            } finally {
                inferenceSlots.release();
            }
            return toGenerationDto(result, target.modelType());
        } catch (IOException e) {
            throw new ServerProcessException("Generation I/O failure", e);
        } finally {
            deleteQuietly(inputFile);
            deleteQuietly(outputFile);
        }
    }

    private JsonNode runGenerationScript(ProjectService.InferenceTarget target, Path inputFile, Path outputFile,
                                         UUID projectId, int maxNewTokens, double temperature) {
        File wrapper = new File(inferWrapperPath);
        List<String> command = new ArrayList<>();
        if (!System.getProperty("os.name").toLowerCase().contains("win")) command.add("bash");
        command.add(wrapper.getAbsolutePath());
        command.add("--model-path"); command.add(target.modelPath());
        command.add("--model-type"); command.add(target.modelType());
        command.add("--model-name"); command.add(target.modelName() == null ? "" : target.modelName());
        command.add("--task-type"); command.add("CAUSAL_LM");
        command.add("--max-new-tokens"); command.add(String.valueOf(maxNewTokens));
        command.add("--temperature"); command.add(String.valueOf(temperature));
        command.add("--in"); command.add(inputFile.toAbsolutePath().toString());
        command.add("--out"); command.add(outputFile.toAbsolutePath().toString());

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        pb.redirectErrorStream(true);
        StringBuilder diag = new StringBuilder();
        Process process;
        try {
            process = pb.start();
        } catch (IOException e) {
            throw new ServerProcessException("Failed to start generation process", e);
        }
        runningGenerations.put(projectId, process);
        try {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (!broadcastIfToken(projectId, line)) {
                        diag.append(line).append('\n');
                    }
                }
                boolean finished = process.waitFor(generationTimeoutSeconds, TimeUnit.SECONDS);
                if (!finished) {
                    process.destroyForcibly();
                    throw new ServerProcessException("Generation timed out after " + generationTimeoutSeconds + "s");
                }
            } catch (IOException e) {
                if (stoppedGenerations.contains(projectId)) return stoppedResult(target.modelType());
                throw new ServerProcessException("Generation process I/O error", e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                process.destroyForcibly();
                throw new ServerProcessException("Generation interrupted", e);
            }
            JsonNode result = readResult(outputFile);
            if (result == null) {
                // A user-stop kills infer.py before it writes the out-file — return a clean "stopped"
                // result (the client keeps its streamed partial), not a 502.
                if (stoppedGenerations.contains(projectId)) return stoppedResult(target.modelType());
                log.warn("Generation produced no result file. Output:\n{}", diag);
                throw new ServerProcessException("Generation produced no result (exit=" + process.exitValue() + ")");
            }
            if (!result.path("ok").asBoolean(false)) {
                String err = result.path("error").asText("Generation failed");
                if ("input".equals(result.path("errorKind").asText("internal"))) {
                    throw new IllegalArgumentException(err);
                }
                log.warn("Generation internal failure for {}: {}", target.modelType(), err);
                throw new ServerProcessException("Generation failed while executing the model");
            }
            return result;
        } finally {
            runningGenerations.remove(projectId, process);
            stoppedGenerations.remove(projectId);
        }
    }

    /** Cancel the in-flight generation for a project (authz-gated). Returns true if one was running. */
    public boolean stopGeneration(@NonNull UUID projectId) {
        projectService.resolveInferenceTarget(projectId); // same participant/org authz as generate (404/403)
        return stopTrackedGeneration(projectId);
    }

    /** Mark stopped + destroy the tracked process. Package-private for unit testing. */
    boolean stopTrackedGeneration(UUID projectId) {
        Process p = runningGenerations.get(projectId);
        if (p == null) return false;          // nothing running → harmless no-op (flag NOT set)
        stoppedGenerations.add(projectId);
        p.destroyForcibly();
        return true;
    }

    /** Synthetic result for a user-stopped generation; the client keeps its streamed partial. */
    JsonNode stoppedResult(String modelType) {
        com.fasterxml.jackson.databind.node.ObjectNode n = objectMapper.createObjectNode();
        n.put("ok", true);
        n.put("modelType", modelType);
        n.put("prompt", "");
        n.put("generatedText", "");
        n.put("tokenCount", 0);
        n.put("finishReason", "stopped");
        return n;
    }

    private GenerationResultDto toGenerationDto(JsonNode r, String modelType) {
        GenerationResultDto dto = new GenerationResultDto();
        dto.setModelType(r.path("modelType").asText(modelType));
        dto.setPrompt(r.path("prompt").asText(""));
        dto.setGeneratedText(r.path("generatedText").asText(""));
        dto.setTokenCount(r.path("tokenCount").asInt());
        dto.setFinishReason(r.path("finishReason").asText("stop"));
        return dto;
    }

    private void deleteQuietly(Path p) {
        if (p == null) return;
        try {
            Files.deleteIfExists(p);
        } catch (IOException e) {
            log.debug("Could not delete temp file {}: {}", p, e.getMessage());
        }
    }
}
