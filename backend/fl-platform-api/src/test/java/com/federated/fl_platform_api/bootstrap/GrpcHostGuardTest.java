package com.federated.fl_platform_api.bootstrap;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * OP-3: a deployed profile with a non-client-reachable grpc-host must be flagged (fail loud) so FL
 * clients aren't handed an address that resolves to their own machine. Pure static — no Spring.
 */
class GrpcHostGuardTest {

    @Test
    void deployedProfileWithNonReachableHost_isFlagged() {
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), "localhost").isPresent());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("production"), "127.0.0.1").isPresent());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), "0.0.0.0").isPresent());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), "::1").isPresent());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), null).isPresent());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), "  LOCALHOST ").isPresent());  // case/trim
    }

    @Test
    void deployedProfileWithRealHost_isOk() {
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("ec2demo"), "10.0.0.5").isEmpty());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("production"), "fedlearn.example.com").isEmpty());
    }

    @Test
    void nonDeployedProfiles_areNeverFlagged() {
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("dev"), "localhost").isEmpty());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of("test"), "localhost").isEmpty());
        assertTrue(BootstrapRunner.grpcHostMisconfig(List.of(), "localhost").isEmpty());
    }
}
