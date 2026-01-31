package com.federated.fl_platform_api.service;

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

    @Value("${python.executable.path}") // Points to run_init_model.bat or .sh
    private String initModelWrapperPath;

    public void initializeModelFile(String modelType, String modelName, String optimizer, String outputPath, int pretrainEpochs) throws IOException, InterruptedException {

        System.out.println("--- Preparing to Execute Model Initializer ---");

        List<String> command = new ArrayList<>();
        String os = System.getProperty("os.name").toLowerCase();

        // Determine script path based on OS
        String scriptPath;
        if (os.contains("win")) {
            // Windows - use .bat file
            scriptPath = initModelWrapperPath.replace(".bat", ".bat");
            command.add(scriptPath);
        } else {
            // Linux/Mac - use .sh file and call with bash
            scriptPath = initModelWrapperPath.replace(".bat", ".sh");
            File scriptFile = new File(scriptPath);
            String absoluteScriptPath = scriptFile.getAbsolutePath();
            command.add("bash");
            command.add(absoluteScriptPath);
        }

        // Add the arguments for the script
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

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(new File("."));
        pb.redirectErrorStream(true);

        System.out.println("--- Starting Model Initializer Process ---");
        System.out.println("Command: " + String.join(" ", pb.command()));

        Process process = pb.start();

        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("[INIT_MODEL_LOG] " + line);
                output.append(line).append("\n");
            }
        }

        int exitCode = process.waitFor();
        System.out.println("--- Model Initializer Process Finished with Exit Code: " + exitCode + " ---");

        if (exitCode != 0) {
            throw new RuntimeException("Model initialization script failed with exit code: " + exitCode +
                    "\nFull Output:\n" + output.toString());
        }

        System.out.println("--- Model File Successfully Created ---");
    }
}