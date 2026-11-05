package com.federated.fl_platform_api;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.fasterxml.jackson.databind.JsonNode;
import java.time.Instant;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;

/** A client credential is usable only through its currently authorized server execution. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class InternalRunAuthorizationIntegrationTest {
    @Autowired TestRestTemplate rest;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired UserRepository users;
    @Autowired RunTokenRegistry tokens;
    @Autowired ConnectionTokenService clients;

    record ActiveRun(Project project, Run run, String token) {}

    private ActiveRun activeRun() {
        String suffix = UUID.randomUUID().toString();
        User owner = users.save(new User("conn-" + suffix, suffix + "@example.com", "test-only-hash"));
        Project project = new Project();
        project.setName("conn-" + suffix);
        project.setUser(owner);
        project.setModelType("CNN");
        project.setModelName("net");
        project.setStatus("RUNNING");
        project.setOrgId(UUID.fromString("00000000-0000-0000-0000-000000000001"));
        project.setVisibility(ProjectVisibility.PRIVATE);
        return newRun(projects.saveAndFlush(project));
    }

    private ActiveRun newRun(Project project) {
        Run run = new Run();
        run.setProjectId(project.getId());
        run.setStrategy("FedAvg");
        run.setRecipeKey("CNN");
        run.setNumRounds(3);
        run.setMinClients(1);
        run.setClientsPerRound(1);
        run.setStatus(RunStatus.RUNNING);
        run.setCreatedAt(Instant.now());
        run = runs.saveAndFlush(run);
        String token = tokens.mint(project.getId(), run.getId());
        run.setInternalTokenHash(tokens.hash(token));
        run = runs.saveAndFlush(run);
        project.setActiveRunId(run.getId());
        return new ActiveRun(projects.saveAndFlush(project), run, token);
    }

    private ResponseEntity<JsonNode> verify(ActiveRun active, String serverToken, String clientToken) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("X-Internal-Key", "test-internal-key");
        headers.set("X-Internal-Run-Token", serverToken);
        return rest.exchange("/api/internal/runs/" + active.project().getId() + "/" + active.run().getId()
                + "/verify-connection-token", HttpMethod.POST,
                new HttpEntity<>("{\"connectionToken\":\"" + clientToken + "\"}", headers), JsonNode.class);
    }

    private String clientToken(ActiveRun active) {
        return clients.mint(new ConnectionTokenService.Claims(99L, active.run().getId(), active.project().getId(),
                2, "localhost:50000", null, "SHARD")).token();
    }

    @Test
    void currentRunningServerResolvesTheSignedClientIdentity() {
        ActiveRun active = activeRun();
        var response = verify(active, active.token(), clientToken(active));
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(99, response.getBody().path("userId").asInt());
        assertEquals(2, response.getBody().path("partitionId").asInt());
        assertEquals(active.run().getId().toString(), response.getBody().path("runId").asText());
    }

    @Test
    void anotherServerRunInTheSameProjectCannotVerifyTheCurrentClientCredential() {
        ActiveRun active = activeRun();
        String other = tokens.mint(active.project().getId(), UUID.randomUUID());
        assertEquals(HttpStatus.FORBIDDEN, verify(active, other, clientToken(active)).getStatusCode());
    }

    @Test
    void anOlderServerCannotVerifyAfterAnotherRunBecomesActive() {
        ActiveRun old = activeRun();
        newRun(old.project());
        assertTrue(tokens.resolve(old.token()).isPresent());
        assertEquals(HttpStatus.CONFLICT, verify(old, old.token(), clientToken(old)).getStatusCode());
    }

    @Test
    void rotatedServerAuthorizationIsRefusedEvenWhenItsRegistryEntryStillExists() {
        ActiveRun active = activeRun();
        active.run().setInternalTokenHash(tokens.hash(tokens.mint(active.project().getId(), active.run().getId())));
        runs.saveAndFlush(active.run());
        assertTrue(tokens.resolve(active.token()).isPresent());
        assertEquals(HttpStatus.FORBIDDEN, verify(active, active.token(), clientToken(active)).getStatusCode());
    }

    @ParameterizedTest
    @EnumSource(value = RunStatus.class, names = {"COMPLETED", "FAILED", "STOPPED"})
    void terminalServerCannotAuthorizeAClient(RunStatus status) {
        ActiveRun active = activeRun();
        active.run().setStatus(status);
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.CONFLICT, verify(active, active.token(), clientToken(active)).getStatusCode());
    }

    @Test
    void startingServerCanAuthenticateAnEarlyClient() {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.STARTING);
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.OK, verify(active, active.token(), clientToken(active)).getStatusCode());
    }

    @Test
    void activeServerStillRejectsAnUnauthenticClientCredential() {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.UNAUTHORIZED, verify(active, active.token(), "invalid-client-token").getStatusCode());
    }
}
