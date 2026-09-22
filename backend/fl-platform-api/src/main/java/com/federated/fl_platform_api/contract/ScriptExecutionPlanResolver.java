package com.federated.fl_platform_api.contract;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fedlearn.contract.v1.ModelTraining;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * Runs fl-runtime/execution_plan.py through its wrapper, the same way the backend runs the other fl-runtime scripts,
 * and parses the plan it prints. Arguments are checked against their vocabularies before anything is spawned, and
 * the initial-model path is passed in {@code --flag=value} form so it can never be read as an option.
 */
@Component
public class ScriptExecutionPlanResolver implements ExecutionPlanResolver {

    private static final Pattern RECIPE_KEY = Pattern.compile("[A-Z][A-Z0-9_]*");
    private static final Pattern STRATEGY = Pattern.compile("[A-Za-z][A-Za-z0-9]*");
    private static final Pattern TRAINING_ARM = Pattern.compile("[A-Z][A-Z_]*");
    private static final ObjectMapper JSON = new ObjectMapper();

    @Value("${python.script.execution-plan.path:../../fl-runtime/run_execution_plan.sh}")
    private String wrapperPath;

    @Value("${app.execution-plan.timeout-seconds:120}")
    private long timeoutSeconds;

    @Override
    public ModelTraining resolve(String recipeKey, String strategy, String trainingArm, Path initialState)
            throws NotRepresentableException, IOException {
        requireMatch("recipe", RECIPE_KEY, recipeKey);
        requireMatch("strategy", STRATEGY, strategy);
        requireMatch("training arm", TRAINING_ARM, trainingArm);
        List<String> command = List.of("bash", new File(wrapperPath).getAbsolutePath(),
                "--recipe", recipeKey, "--strategy", strategy, "--training-arm", trainingArm,
                "--initial-state=" + initialState.toAbsolutePath());
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectError(ProcessBuilder.Redirect.DISCARD);
        Process process = pb.start();
        String stdout;
        try (InputStream in = process.getInputStream()) {
            stdout = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        try {
            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IOException("the execution-plan resolver timed out after " + timeoutSeconds + "s");
            }
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while resolving the execution plan", e);
        }
        if (process.exitValue() != 0) {
            throw new IOException("the execution-plan resolver exited " + process.exitValue());
        }
        return parse(lastLine(stdout));
    }

    private static void requireMatch(String what, Pattern pattern, String value) {
        if (value == null || !pattern.matcher(value).matches()) {
            throw new IllegalArgumentException("refusing to resolve an execution plan for " + what + " " + value);
        }
    }

    /** The plan is the last line printed; library chatter before it is ignored. */
    private static String lastLine(String stdout) throws IOException {
        String[] lines = stdout.strip().split("\n");
        String last = lines[lines.length - 1].strip();
        if (last.isEmpty()) {
            throw new IOException("the execution-plan resolver printed nothing");
        }
        return last;
    }

    private static ModelTraining parse(String line) throws NotRepresentableException, IOException {
        JsonNode out = JSON.readTree(line);
        if (!out.path("representable").isBoolean()) {
            throw new IOException("the execution-plan resolver printed no verdict");
        }
        if (!out.get("representable").asBoolean()) {
            throw new NotRepresentableException(out.path("reason").asText("no reason given"));
        }
        ModelTraining.Builder plan = ModelTraining.newBuilder();
        try {
            JsonFormat.parser().merge(JSON.writeValueAsString(out.path("modelTraining")), plan);
        } catch (InvalidProtocolBufferException e) {
            throw new IOException("the execution-plan resolver printed an unreadable plan", e);
        }
        return plan.build();
    }
}
