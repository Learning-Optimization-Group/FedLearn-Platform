package com.federated.fl_platform_api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class ProjectResultRunSelectionIntegrationTest {
    @Autowired TestRestTemplate rest;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired RoundResultRepository results;
    @Autowired UserRepository users;
    @Autowired PasswordEncoder passwords;
    @Autowired ObjectMapper json;

    record Fixture(User owner, Project project, Run previous, Run current) {}

    private User user() {
        String suffix = UUID.randomUUID().toString();
        return users.saveAndFlush(new User("result-" + suffix, suffix + "@example.invalid",
                passwords.encode("test-only-password")));
    }

    private Run run(Project project, int rounds) {
        Run value = new Run();
        value.setProjectId(project.getId());
        value.setRecipeKey("CNN");
        value.setStrategy("FedAvg");
        value.setStatus(RunStatus.COMPLETED);
        value.setNumRounds(rounds);
        value.setMinClients(2);
        value.setClientsPerRound(2);
        value.setCreatedAt(Instant.now());
        return runs.saveAndFlush(value);
    }

    private void metric(Project project, UUID runId, int round, double accuracy) {
        RoundResult value = new RoundResult();
        value.setProject(project);
        value.setRunId(runId);
        value.setServerRound(round);
        value.setLoss(1.0 - accuracy);
        value.setAccuracy(accuracy);
        value.setGpuUtilization(0.0);
        results.saveAndFlush(value);
    }

    private Fixture fixture(boolean currentHasResults) {
        User owner = user();
        Project project = new Project();
        project.setName("result-scope-" + UUID.randomUUID());
        project.setUser(owner);
        project.setOrgId(UUID.fromString("00000000-0000-0000-0000-000000000001"));
        project.setModelType("CNN");
        project.setModelName("net");
        project.setStatus("COMPLETED");
        project = projects.saveAndFlush(project);
        Run previous = run(project, 3);
        for (int round = 1; round <= 3; round++) metric(project, previous.getId(), round, 0.30 + round / 100.0);
        Run current = run(project, 1);
        if (currentHasResults) metric(project, current.getId(), 1, 0.90);
        project.setActiveRunId(current.getId());
        return new Fixture(owner, projects.saveAndFlush(project), previous, current);
    }

    private HttpHeaders login(User user) {
        ResponseEntity<String> response = rest.postForEntity("/api/auth/login",
                Map.of("username", user.getUsername(), "password", "test-only-password"), String.class);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        String cookie = response.getHeaders().getFirst(HttpHeaders.SET_COOKIE);
        assertNotNull(cookie);
        HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.COOKIE, cookie.split(";", 2)[0]);
        return headers;
    }

    private ResponseEntity<String> read(Fixture fixture, String query, User caller) {
        return rest.exchange("/api/projects/" + fixture.project().getId() + "/results" + query,
                HttpMethod.GET, new HttpEntity<>(login(caller)), String.class);
    }

    @Test
    void currentRunDoesNotInheritAnEarlierRunsLongerCurve() throws Exception {
        Fixture fixture = fixture(true);
        ResponseEntity<String> response = read(fixture, "", fixture.owner());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        JsonNode rows = json.readTree(response.getBody());
        assertEquals(1, rows.size());
        assertEquals(1, rows.get(0).get("serverRound").asInt());
        assertEquals(0.90, rows.get(0).get("accuracy").asDouble(), 1e-12);
    }

    @Test
    void explicitHistoricalRunKeepsItsOwnThreeRounds() throws Exception {
        Fixture fixture = fixture(true);
        ResponseEntity<String> response = read(fixture, "?runId=" + fixture.previous().getId(), fixture.owner());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        JsonNode rows = json.readTree(response.getBody());
        assertEquals(3, rows.size());
        assertEquals(0.33, rows.get(2).get("accuracy").asDouble(), 1e-12);
    }

    @Test
    void newRunWithoutResultsDoesNotDisplayEarlierSuccess() throws Exception {
        Fixture fixture = fixture(false);
        ResponseEntity<String> response = read(fixture, "", fixture.owner());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals(0, json.readTree(response.getBody()).size());
    }

    @Test
    void unknownRunSelectionIsNotAnUnscopedProjectRead() {
        Fixture fixture = fixture(true);
        assertEquals(HttpStatus.NOT_FOUND,
                read(fixture, "?runId=" + UUID.randomUUID(), fixture.owner()).getStatusCode());
    }

    @Test
    void anotherProjectsRunCannotBeSelected() {
        Fixture fixture = fixture(true);
        Fixture another = fixture(true);
        assertEquals(HttpStatus.NOT_FOUND,
                read(fixture, "?runId=" + another.current().getId(), fixture.owner()).getStatusCode());
    }

    @Test
    void nonOwnerCannotReadCurrentOrHistoricalResults() {
        Fixture fixture = fixture(true);
        User another = user();
        assertEquals(HttpStatus.FORBIDDEN, read(fixture, "", another).getStatusCode());
        assertEquals(HttpStatus.FORBIDDEN,
                read(fixture, "?runId=" + fixture.previous().getId(), another).getStatusCode());
    }

    @Test
    void aProjectWithoutACurrentRunReadsOnlyItsUnboundLegacyRows() throws Exception {
        Fixture fixture = fixture(true);
        fixture.project().setActiveRunId(null);
        projects.saveAndFlush(fixture.project());
        metric(fixture.project(), null, 7, 0.20);
        ResponseEntity<String> response = read(fixture, "", fixture.owner());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        JsonNode rows = json.readTree(response.getBody());
        assertEquals(1, rows.size());
        assertEquals(7, rows.get(0).get("serverRound").asInt());
    }

    @Test
    void malformedRunSelectionIsRejected() {
        Fixture fixture = fixture(true);
        assertEquals(HttpStatus.BAD_REQUEST,
                read(fixture, "?runId=not-a-run", fixture.owner()).getStatusCode());
    }
}
