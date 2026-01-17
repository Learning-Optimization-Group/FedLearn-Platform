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

    @Value("${python.executable.path}") // Points to run_init_model.bat
    private String initModelWrapperPath;

    // In ModelInitializer.java

    public void initializeModelFile(String modelType, String modelName, String optimizer, String outputPath, int pretrainEpochs) throws IOException, InterruptedException {

        // --- THIS IS THE CRUCIAL FIX ---
        // Create a File object from the relative path and get its absolute path.
        // This ensures that no matter what, we are calling the script with its full, unambiguous path.
        File scriptFile = new File(initModelWrapperPath);
        ProcessBuilder pb;
        String absoluteScriptPath = scriptFile.getAbsolutePath();

        System.out.println("--- Preparing to Execute Model Initializer ---");

        List<String> command = new ArrayList<>();
        String os = System.getProperty("os.name").toLowerCase();

        if (!os.contains("win")) {
            // On Windows, the command is just the .bat file
            command.add("bash");

        }

        // Add the arguments for the script
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

        pb = new ProcessBuilder(command);
//        ProcessBuilder pb = new ProcessBuilder(
//                "bash",
//                absoluteScriptPath, // Use the absolute path here
//                "--model", modelType,
//                "--out", outputPath,
//                "--pretrain-epochs", String.valueOf(pretrainEpochs)
//        );

        // Set the working directory to ensure all relative paths inside the script are correct.
        pb.directory(new File("."));
        pb.redirectErrorStream(true);

        System.out.println("--- Starting Model Initializer Process ---");
        System.out.println("Command: " + String.join(" ", pb.command()));

        Process process = pb.start();

        // ... (rest of the logging and waitFor logic is the same)
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