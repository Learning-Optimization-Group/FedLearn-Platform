package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * BA-4 done-when #1: ProjectService must set and compare project status through the
 * {@link com.federated.fl_platform_api.model.ProjectStatus} enum, never a raw String literal —
 * a drifted literal is exactly what let a failed run keep showing {@code RUNNING}. This guard reads
 * the service source and fails if any raw status literal creeps back in.
 */
class ProjectServiceStatusLiteralGuardTest {

    private static final Path SOURCE =
            Path.of("src/main/java/com/federated/fl_platform_api/service/ProjectService.java");

    @Test
    void projectService_setsAndComparesStatusViaTheEnum_notRawLiterals() throws IOException {
        String code = Files.readString(SOURCE);

        // No status set from a raw String literal — it must go through ProjectStatus.X.name().
        assertThat(code)
                .as("ProjectService must not call setStatus(\"...\") with a raw literal (BA-4)")
                .doesNotContain("setStatus(\"");

        // No raw-literal status comparison either (e.g. \"RUNNING\".equals(project.getStatus())).
        for (String status : List.of("CREATED", "RUNNING", "STOPPED", "COMPLETED", "FAILED", "INITIALIZING")) {
            assertThat(code)
                    .as("ProjectService must not compare against the raw status literal \"%s\" (BA-4)", status)
                    .doesNotContain("\"" + status + "\".equals");
        }
    }
}
