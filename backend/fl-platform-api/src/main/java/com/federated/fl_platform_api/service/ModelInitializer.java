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

@Component
public class ModelInitializer {

    private static final Logger log = LoggerFactory.getLogger(ModelInitializer.class);

    @Value("${python.executable.path}")
    private String initModelWrapperPath;

    public void initializeModelFile(String modelType, String modelName, String optimizer,
                                    String outputPath, int pretrainEpochs)
            throws IOException, InterruptedException {

        File scriptFile = new File(initModelWrapperPath);
        String absoluteScriptPath = scriptFile.getAbsolutePath();

        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        List<String> command = buildInitCommand(
                modelType, modelName, optimizer, outputPath, pretrainEpochs, absoluteScriptPath, isWindows);

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        pb.redirectErrorStream(true);

        log.debug("Spawning model initializer for {}/{} → {}", modelType, modelName, outputPath);
        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                log.debug("[init_model] {}", line);
                output.append(line).append('\n');
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            // Caller wraps in ServerProcessException for the API layer; we throw
            // it here too so a direct caller (test, etc.) gets a typed failure.
            throw new ServerProcessException(
                    "Model initialization script failed (exit=" + exitCode + ")\nOutput:\n" + output);
        }

        log.info("Model file initialized at {}", outputPath);
    }

    /** Build the init_model launch command. LLM_LORA carries --aggregation FFA_LORA. */
    static List<String> buildInitCommand(String modelType, String modelName, String optimizer,
                                         String outputPath, int pretrainEpochs, String absoluteScriptPath,
                                         boolean isWindows) {
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
        }
        return command;
    }
}
