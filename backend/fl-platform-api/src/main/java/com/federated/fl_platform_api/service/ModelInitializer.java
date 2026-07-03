package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.exception.ServerProcessException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Component
public class ModelInitializer {

    private static final Logger log = LoggerFactory.getLogger(ModelInitializer.class);

    @Value("${python.executable.path}")
    private String initModelWrapperPath;

    // BA-1: a hung init must not block the request thread (holding a DB connection) forever.
    @Value("${python.script.init-model.timeout-seconds:300}")
    private long initTimeoutSeconds;

    public void initializeModelFile(String modelType, String modelName, String optimizer,
                                    String outputPath, int pretrainEpochs, String taskType)
            throws IOException, InterruptedException {

        File scriptFile = new File(initModelWrapperPath);
        String absoluteScriptPath = scriptFile.getAbsolutePath();

        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        List<String> command = buildInitCommand(
                modelType, modelName, optimizer, outputPath, pretrainEpochs, taskType, absoluteScriptPath, isWindows);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        pb.redirectErrorStream(true);

        log.debug("Spawning model initializer for {}/{} → {}", modelType, modelName, outputPath);
        Process process = pb.start();

        // Drain stdout on a daemon thread so the child never blocks on a full pipe buffer, and so the
        // main thread can enforce a timeout instead of blocking in readLine() until the child exits.
        final StringBuilder output = new StringBuilder();
        Thread reader = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    log.debug("[init_model] {}", line);
                    synchronized (output) {
                        output.append(line).append('\n');
                    }
                }
            } catch (IOException e) {
                log.warn("Failed reading init_model output: {}", e.getClass().getSimpleName());
            }
        }, "init-model-stdout");
        reader.setDaemon(true);
        reader.start();

        boolean finished = process.waitFor(initTimeoutSeconds, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            throw new ServerProcessException(
                    "Model initialization timed out after " + initTimeoutSeconds + "s for " + outputPath
                            + " (killed the process)");
        }
        reader.join(5000);   // let the reader drain any buffered output before we read it

        int exitCode = process.exitValue();
        if (exitCode != 0) {
            String captured;
            synchronized (output) {
                captured = output.toString();
            }
            // Caller wraps in ServerProcessException for the API layer; we throw it here too so a
            // direct caller (test, etc.) gets a typed failure.
            throw new ServerProcessException(
                    "Model initialization script failed (exit=" + exitCode + ")\nOutput:\n" + captured);
        }

        log.info("Model file initialized at {}", outputPath);
    }

    /** Build the init_model launch command. LLM_LORA carries --aggregation FFA_LORA and --task-type. */
    static List<String> buildInitCommand(String modelType, String modelName, String optimizer,
                                         String outputPath, int pretrainEpochs, String taskType,
                                         String absoluteScriptPath, boolean isWindows) {
        List<String> command = new ArrayList<>();
        if (!isWindows) {
            command.add("bash");
        }
        command.add(absoluteScriptPath);
        command.add("--model-type");
        command.add(modelType);
        command.add("--model-name");
        command.add(modelName);
        command.add("--optimizer");
        command.add(optimizer);
        command.add("--out");
        command.add(outputPath);
        command.add("--pretrain-epochs");
        command.add(String.valueOf(pretrainEpochs));
        if ("LLM_LORA".equalsIgnoreCase(modelType)) {
            command.add("--aggregation");
            command.add("FFA_LORA");
            command.add("--task-type");
            command.add(taskType == null || taskType.isBlank() ? "SEQ_CLASSIFICATION" : taskType);
        }
        return command;
    }
}
