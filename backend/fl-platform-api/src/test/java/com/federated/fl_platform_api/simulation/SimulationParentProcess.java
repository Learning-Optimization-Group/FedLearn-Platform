package com.federated.fl_platform_api.simulation;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

/** Isolated backend-owner process for the abrupt-death integration control. */
public final class SimulationParentProcess {
    public static void main(String[] args) throws Exception {
        SimulationProperties properties = new SimulationProperties();
        properties.setEnabled(true); properties.setInterpreter(Path.of(args[0])); properties.setSourceRoot(Path.of(args[1]));
        properties.setCache(Path.of(args[2])); properties.setStorage(Path.of(args[3])); properties.setReportSha256(args[4]);
        try (var runner = new SimulationProcessRunner(properties)) {
            var child = runner.launch(UUID.fromString(args[5]), SimulationJson.stored(Files.readString(Path.of(args[6]))));
            Files.writeString(Path.of(args[7]), Long.toString(child.process().pid()));
            Thread.sleep(30000);
        }
    }
}
