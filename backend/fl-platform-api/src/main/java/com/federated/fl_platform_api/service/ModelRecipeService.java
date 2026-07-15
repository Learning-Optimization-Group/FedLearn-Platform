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
 * <p>{@code recipes.py} is the single source of truth for a model type's interactive input kind and
 * class labels. It is loaded once (lazily, on first request) and cached in memory for the JVM
 * lifetime — recipes are static config, so no refresh is needed. There is deliberately NO hardcoded
 * Java fallback (DA-10): a fallback duplicate had already drifted from the catalog (it was missing
 * BLOOD_CNN/LLM_LORA), and since the app spawns python for all training/inference anyway, a broken
 * {@code recipes.py} should surface loudly rather than be masked by stale data. A load failure
 * throws {@link IllegalStateException} and is not cached, so a transient problem recovers on retry.
 */
@Service
public class ModelRecipeService {

    private static final Logger log = LoggerFactory.getLogger(ModelRecipeService.class);

    /** Recipe discovery is a quick metadata dump (no torch); a short cap is plenty. */
    private static final long PROCESS_TIMEOUT_SECONDS = 30;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${python.script.recipes.path:../../fl-runtime/run_recipes.sh}")
    private String recipesWrapperPath;

    /** Lazily-populated cache; once set it is never re-read. Volatile for safe publication. */
    private volatile List<ModelRecipeDto> cache;

    /** The full catalog, loading + caching on first call. Never null/empty; throws
     * {@link IllegalStateException} if recipes.py can't be loaded (the failure is not cached). */
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
        List<ModelRecipeDto> parsed;
        try {
            parsed = runDescribe();
        } catch (Exception e) {
            // DA-10: recipes.py is the single source of truth — no hardcoded fallback (it had drifted,
            // missing BLOOD_CNN/LLM_LORA). Fail loud; getRecipes() only assigns the cache on a clean
            // return, so this failure is NOT cached and a transient python/script problem recovers on
            // the next request rather than being masked by a stale duplicate catalog.
            throw new IllegalStateException(
                    "Failed to load the model-recipe catalog from " + recipesWrapperPath
                            + " (recipes.py is the single source of truth): " + e.getMessage(), e);
        }
        if (parsed == null || parsed.isEmpty()) {
            throw new IllegalStateException(
                    "recipes.py --describe returned no recipes from " + recipesWrapperPath);
        }
        log.info("Loaded {} model recipes from {}", parsed.size(), recipesWrapperPath);
        return parsed;
    }

    /** Spawn {@code recipes.py --describe} and parse its JSON. Package-private/overridable as a test seam. */
    protected List<ModelRecipeDto> runDescribe() throws IOException, InterruptedException {
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

}
