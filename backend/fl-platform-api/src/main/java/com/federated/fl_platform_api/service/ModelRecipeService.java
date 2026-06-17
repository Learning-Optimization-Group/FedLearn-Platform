package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.dto.ModelRecipeDto;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * Loads and caches the model-recipe catalog from the framework's {@code recipes.py}
 * (via the {@code run_recipes.sh} wrapper), spawning the same way
 * {@link InferenceService}/{@link ModelInitializer} spawn their wrappers.
 *
 * <p>The catalog is the single source of truth for a model type's interactive
 * input kind and class labels. It is loaded once (lazily, on first request) and
 * cached in memory for the JVM lifetime — recipes are static config, so no
 * refresh is needed. If the script is missing or its output can't be parsed, we
 * log a WARNING and fall back to a hardcoded catalog so the app still boots and
 * inference keeps working for the built-in CNN/MLP/Transformer/Pneumonia types.
 */
@Service
public class ModelRecipeService {

    private static final Logger log = LoggerFactory.getLogger(ModelRecipeService.class);

    /** Recipe discovery is a quick metadata dump (no torch); a short cap is plenty. */
    private static final long PROCESS_TIMEOUT_SECONDS = 30;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${python.script.recipes.path:src/main/resources/scripts/run_recipes.sh}")
    private String recipesWrapperPath;

    /** Lazily-populated cache; once set it is never re-read. Volatile for safe publication. */
    private volatile List<ModelRecipeDto> cache;

    /** The full catalog, loading + caching on first call. Never null; never empty. */
    public List<ModelRecipeDto> getRecipes() {
        List<ModelRecipeDto> local = cache;
        if (local != null) {
            return local;
        }
        synchronized (this) {
            if (cache == null) {
                cache = loadRecipes();
            }
            return cache;
        }
    }

    /** Look up a recipe by key, case-insensitively. Empty if not found/unknown. */
    public Optional<ModelRecipeDto> findByKey(String key) {
        if (key == null) {
            return Optional.empty();
        }
        String wanted = key.toUpperCase(Locale.ROOT);
        for (ModelRecipeDto r : getRecipes()) {
            if (r.key() != null && r.key().toUpperCase(Locale.ROOT).equals(wanted)) {
                return Optional.of(r);
            }
        }
        return Optional.empty();
    }

    // ─── loading ──────────────────────────────────────────────────────────────

    private List<ModelRecipeDto> loadRecipes() {
        try {
            List<ModelRecipeDto> parsed = runDescribe();
            if (parsed != null && !parsed.isEmpty()) {
                log.info("Loaded {} model recipes from {}", parsed.size(), recipesWrapperPath);
                return parsed;
            }
            log.warn("Recipe script returned no recipes; using built-in fallback catalog.");
        } catch (Exception e) {
            log.warn("Could not load model recipes from {} ({}); using built-in fallback catalog.",
                    recipesWrapperPath, e.getMessage());
        }
        return fallbackRecipes();
    }

    private List<ModelRecipeDto> runDescribe() throws IOException, InterruptedException {
        File wrapper = new File(recipesWrapperPath);
        String absWrapper = wrapper.getAbsolutePath();

        List<String> command = new ArrayList<>();
        if (!System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("win")) {
            command.add("bash");
        }
        command.add(absWrapper);
        command.add("--describe");

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        // Keep stderr separate: any wrapper/torch log noise must not corrupt the
        // JSON we read off stdout.
        pb.redirectError(ProcessBuilder.Redirect.DISCARD);

        log.debug("Spawning recipe discovery: {}", command);
        Process process = pb.start();

        String stdout;
        try (InputStream in = process.getInputStream()) {
            stdout = readAll(in);
        }

        boolean finished = process.waitFor(PROCESS_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            throw new IOException("recipe discovery timed out after " + PROCESS_TIMEOUT_SECONDS + "s");
        }
        if (process.exitValue() != 0) {
            throw new IOException("recipe discovery exited with code " + process.exitValue());
        }
        if (stdout.isBlank()) {
            throw new IOException("recipe discovery produced no output");
        }
        return objectMapper.readValue(stdout, new TypeReference<List<ModelRecipeDto>>() {});
    }

    private static String readAll(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader =
                     new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }

    // ─── fallback ───────────────────────────────────────────────────────────────
    //
    // Mirrors the prior hardcoded ProjectService switch so behavior is unchanged
    // when recipes.py is unavailable: CNN=image/CIFAR-10, MLP=vector/[Normal,
    // Abnormal], TRANSFORMER=text (no fixed labels), PNEUMONIA_CNN=image/[NORMAL,
    // PNEUMONIA].

    private List<ModelRecipeDto> fallbackRecipes() {
        return List.of(
                new ModelRecipeDto(
                        "CNN", "CNN (CIFAR-10)", "image",
                        List.of("airplane", "automobile", "bird", "cat", "deer",
                                "dog", "frog", "horse", "ship", "truck"),
                        List.of(), List.of()),
                new ModelRecipeDto(
                        "MLP", "MLP", "vector",
                        List.of("Normal", "Abnormal"),
                        List.of(), List.of()),
                new ModelRecipeDto(
                        "TRANSFORMER", "Transformer", "text",
                        List.of(),
                        List.of(), List.of()),
                new ModelRecipeDto(
                        "PNEUMONIA_CNN", "Pneumonia CNN", "image",
                        List.of("NORMAL", "PNEUMONIA"),
                        List.of(), List.of())
        );
    }
}
