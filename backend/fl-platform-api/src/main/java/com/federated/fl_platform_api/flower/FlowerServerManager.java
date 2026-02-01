package com.federated.fl_platform_api.flower;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.WebSocketService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Component
public class FlowerServerManager {

    // This property points to the run_fl_server wrapper script
    @Value("${python.script.fl-server.path}")
    private String flServerWrapperPath;

    // A map to keep track of the running server processes for cleanup
    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    @Autowired
    private WebSocketService logBroadcaster;

    /**
     * Starts a dedicated Flower server process for a given project.
     */
    public int startServerForProject(Project project, boolean isPretrained, String strategy, Integer numRounds, Integer minClients) throws IOException, InterruptedException {

        stopServerForProject(project.getId());
        Thread.sleep(2000);

        int freePort = findFreePort();

        System.out.println("--- Preparing to Start Flower Server ---");

        List<String> command = new ArrayList<>();
        String os = System.getProperty("os.name").toLowerCase();

        // Determine script path based on OS
        String scriptPath;
        if (os.contains("win")) {
            // Windows - use .bat file
            scriptPath = flServerWrapperPath.replace(".sh", ".bat");
            command.add(scriptPath);
        } else {
            // Linux/Mac - use .sh file and call with bash
            scriptPath = flServerWrapperPath.replace(".bat", ".sh");
            File scriptFile = new File(scriptPath);
            String absoluteScriptPath = scriptFile.getAbsolutePath();
            command.add("bash");
            command.add(absoluteScriptPath);
        }

        // Add the arguments for the script
        command.add("--project-id");
        command.add(project.getId().toString());
        command.add("--model-path");
        command.add(project.getModelPath());
        command.add("--port");
        command.add(String.valueOf(freePort));
        command.add("--strategy");
        command.add(strategy);
        command.add("--num-rounds");
        command.add(String.valueOf(numRounds));
        command.add("--min-clients");
        command.add(String.valueOf(minClients));
        command.add("--model-type");
        command.add(project.getModelType());
        command.add("--model-name");
        command.add(project.getModelName());

        if (!isPretrained) {
            command.add("--pretrain");
        }

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        pb.directory(new File("."));

        System.out.println("Executing command: " + String.join(" ", pb.command()));

        Process process = pb.start();
        runningServers.put(project.getId(), process);

        // --- Asynchronous output reader AND process health check ---
        final StringBuilder startupOutput = new StringBuilder();
        final var errorOccurred = new boolean[]{false};

        Thread outputReaderThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("[FL_SERVER_LOG " + project.getId() + "] " + line);
                    logBroadcaster.sendLogs(project.getId(), line);
                    startupOutput.append(line).append("\n");
                }
            } catch (IOException e) {
                System.err.println("Error reading output from Flower server process for project " + project.getId());
                errorOccurred[0] = true;
                logBroadcaster.sendLogs(project.getId(), "ERROR: " + e);
                e.printStackTrace();
            }
        });
        outputReaderThread.setDaemon(true);
        outputReaderThread.start();

        // Wait for a short period to see if the process exits immediately
        boolean exited = process.waitFor(3, TimeUnit.SECONDS);

        if (exited || errorOccurred[0]) {
            outputReaderThread.join(1000);
            throw new RuntimeException("Flower server process failed to start. Exit code: " + process.exitValue() +
                    "\nFull Output:\n" + startupOutput);
        }

        System.out.println("Started Flower server for project " + project.getName() + " on port " + freePort);
        return freePort;
    }

    public boolean stopServerForProject(UUID projectId) {
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            System.out.println("Stopping Flower server for project: " + projectId);
            process.destroyForcibly();
            runningServers.remove(projectId);
            return true;
        }
        System.out.println("No running server found for project: " + projectId);
        return false;
    }

    private int findFreePort() {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            if (serverSocket != null) {
                return serverSocket.getLocalPort();
            }
        } catch (IOException e) {
            throw new IllegalStateException("Could not find a free TCP/IP port", e);
        }
        throw new IllegalStateException("Could not find a free TCP/IP port");
    }

    public boolean isServerRunning(UUID projectId) {
        Process p = runningServers.get(projectId);
        return (p != null && p.isAlive());
    }
}