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

    // This property points to the run_fl_server.bat wrapper script
    @Value("${python.script.fl-server.path}")
    private String flServerWrapperPath;



    // A map to keep track of the running server processes for cleanup
    private final Map<UUID, Process> runningServers = new ConcurrentHashMap<>();

    @Autowired
    private WebSocketService logBroadcaster;
    /**
     * Starts a dedicated Flower server process for a given project.
     *
     * @param project The project entity containing details like ID and model path.
     * @return The network port on which the server was started.
     * @throws IOException If the process cannot be started.
     * @throws InterruptedException If the thread is interrupted while waiting.
     * @throws RuntimeException If the server process fails to start and exits prematurely.
     */
    public int startServerForProject(Project project,  boolean isPretrained, String strategy, Integer numRounds) throws IOException, InterruptedException {

        stopServerForProject(project.getId());
        Thread.sleep(2000);

        int freePort = findFreePort();
        File scriptFile = new File(flServerWrapperPath);
        String absoluteScriptPath = scriptFile.getAbsolutePath();
        ProcessBuilder pb;
        System.out.println("absoluteScriptPath - "+absoluteScriptPath);
        List<String> command = new ArrayList<>();
        String os = System.getProperty("os.name").toLowerCase();

        if (!os.contains("win")) {
            // On Windows, the command is just the .bat file
            command.add("bash");

        }

        // Add the arguments for the script
        command.add(absoluteScriptPath);
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
        command.add("--model-type");
        command.add(project.getModelType());
        command.add("--model-name");
        command.add(project.getModelName());

        pb = new ProcessBuilder(command);
//         pb = new ProcessBuilder(
//                 "bash",
//                absoluteScriptPath,
//                "--project-id", project.getId().toString(),
//                "--model-path", project.getModelPath(),
//                "--port", String.valueOf(freePort)
//        );


        if (!isPretrained) {
            pb.command().add("--pretrain");
        }


        System.out.println("--- Preparing to Start Flower Server ---");
        System.out.println("Executing command: " + String.join(" ", pb.command()));

        // Best practice: redirect error stream and set working directory
        pb.redirectErrorStream(true);
        pb.directory(new File("."));

        Process process = pb.start();
        runningServers.put(project.getId(), process);

        // --- Asynchronous output reader AND process health check ---
        final StringBuilder startupOutput = new StringBuilder();
        final var errorOccurred = new boolean[]{false}; // Using an array to be modifiable from lambda

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
        outputReaderThread.setDaemon(true); // Allow the JVM to exit even if this thread is running
        outputReaderThread.start();

        // Wait for a short period (e.g., 3 seconds) to see if the process exits immediately.
        // A healthy server should NOT exit.
        boolean exited = process.waitFor(3, TimeUnit.SECONDS);

        if (exited || errorOccurred[0]) {
            // The process terminated, which means it crashed.
            outputReaderThread.join(1000); // Wait a moment for the output reader to catch up
            throw new RuntimeException("Flower server process failed to start. Exit code: " + process.exitValue() +
                    "\nFull Output:\n" + startupOutput);
        }

        // If we get here, the process is still running after the timeout, so we assume it started successfully.
        System.out.println("Started Flower server for project " + project.getName() + " on port " + freePort);
        return freePort;
    }

    public boolean stopServerForProject(UUID projectId) {
        Process process = runningServers.get(projectId);
        if (process != null && process.isAlive()) {
            System.out.println("Stopping Flower server for project: " + projectId);
            process.destroyForcibly(); // Use destroyForcibly() for a more immediate stop
            runningServers.remove(projectId);
            return true;
        }
        System.out.println("No running server found for project: " + projectId);
        return false;
    }

    /**
     * Finds a free TCP port on the local machine.
     *
     * @return An available TCP port number.
     */
    private int findFreePort() {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            // port 0 tells the OS to assign an ephemeral (temporary) port.
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