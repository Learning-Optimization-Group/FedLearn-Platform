package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;

import java.util.Map;
import java.util.UUID;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class SimulationApiIntegrationTest {
    @Autowired TestRestTemplate http;
    @Autowired UserRepository users;
    @Autowired PasswordEncoder passwords;
    @Autowired SimulationRunRepository runs;

    String cookie(PlatformRole role) {
        String name = "lab-" + UUID.randomUUID();
        User user = new User(name, name + "@example.com", passwords.encode("Password1!"));
        user.setPlatformRole(role);
        users.save(user);
        ResponseEntity<JsonNode> login = http.postForEntity("/api/auth/login",
                Map.of("username", name, "password", "Password1!"), JsonNode.class);
        assertThat(login.getStatusCode()).isEqualTo(HttpStatus.OK);
        return login.getHeaders().getFirst(HttpHeaders.SET_COOKIE).split(";")[0];
    }

    ResponseEntity<JsonNode> request(String cookie, HttpMethod method, String path, String body) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        if (cookie != null) headers.set(HttpHeaders.COOKIE, cookie);
        return http.exchange("/api/simulations" + path, method,
                new HttpEntity<>(body, headers), JsonNode.class);
    }

    @Test void anonymousAndOrdinaryUsersCannotUseAnyLabEndpoint() {
        String ordinary = cookie(PlatformRole.USER);
        for (String path : new String[]{"/workloads", "/runs", "/runs/" + UUID.randomUUID(),
                "/runs/" + UUID.randomUUID() + "/record"}) {
            assertThat(request(null, HttpMethod.GET, path, null).getStatusCode())
                    .as("anonymous " + path).isEqualTo(HttpStatus.UNAUTHORIZED);
            assertThat(request(ordinary, HttpMethod.GET, path, null).getStatusCode())
                    .as("ordinary user " + path).isEqualTo(HttpStatus.FORBIDDEN);
        }
        for (String path : new String[]{"/resolve", "/runs", "/runs/" + UUID.randomUUID() + "/stop"}) {
            assertThat(request(null, HttpMethod.POST, path, "{\"choices\":{}}").getStatusCode())
                    .isEqualTo(HttpStatus.UNAUTHORIZED);
            assertThat(request(ordinary, HttpMethod.POST, path, "{\"choices\":{}}").getStatusCode())
                    .isEqualTo(HttpStatus.FORBIDDEN);
        }
    }

    @Test void unconfiguredWorkloadIsExplicitAndCannotLaunch() {
        for (PlatformRole role : new PlatformRole[]{PlatformRole.PROJECT_OWNER, PlatformRole.PLATFORM_ADMIN}) {
            String cookie = cookie(role);
            var response = request(cookie, HttpMethod.GET, "/workloads", null);
            assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
            JsonNode workload = response.getBody().path("workloads").path(0);
            assertThat(workload.path("id").asText()).isEqualTo("nih-qualified");
            assertThat(workload.path("ready").asBoolean(true)).isFalse();
            assertThat(workload.path("reason").asText()).isNotBlank();
            assertThat(workload.path("limits").path("deadlineSeconds").asInt()).isEqualTo(180);
            assertThat(response.getBody().toString()).doesNotContain("/Users/", "PYTHONPATH", "password");
            assertThat(request(cookie, HttpMethod.POST, "/runs", "{\"choices\":{}}").getStatusCode())
                    .isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
            assertThat(request(cookie, HttpMethod.GET, "/runs", null).getBody().path("content").size()).isZero();
        }
    }

    @Test void strictWrapperDoesNotAcceptDuplicateOrUnknownFieldsOrOversizedBodies() {
        String owner = cookie(PlatformRole.PROJECT_OWNER);
        for (String body : new String[]{"{}", "[]", "null", "{\"choices\":[]}",
                "{\"choices\":{},\"command\":\"anything\"}", "{\"choices\":{},\"choices\":{}}",
                "{\"choices\":{\"rounds\":1,\"rounds\":2}}", "{\"choices\":{\"seed\":NaN}}",
                "{\"choices\":{\"head\":\"" + "a".repeat(70000) + "\"}}"}) {
            assertThat(request(owner, HttpMethod.POST, "/resolve", body).getStatusCode())
                    .as(body.substring(0, Math.min(80, body.length()))).isEqualTo(HttpStatus.BAD_REQUEST);
        }
    }

    @Test void paginationIsBoundedAndUnknownRunsDoNotLeak() {
        String owner = cookie(PlatformRole.PROJECT_OWNER);
        var page = request(owner, HttpMethod.GET, "/runs?page=0&size=20", null);
        assertThat(page.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(page.getBody().path("content").isArray()).isTrue();
        assertThat(page.getBody().path("page").asInt(-1)).isZero();
        for (String query : new String[]{"?page=-1", "?size=0", "?size=101", "?scope=platform"}) {
            assertThat(request(owner, HttpMethod.GET, "/runs" + query, null).getStatusCode())
                    .isIn(HttpStatus.BAD_REQUEST, HttpStatus.FORBIDDEN);
        }
        UUID absent = UUID.randomUUID();
        assertThat(request(owner, HttpMethod.GET, "/runs/" + absent, null).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        assertThat(request(owner, HttpMethod.POST, "/runs/" + absent + "/stop", "{}").getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        assertThat(request(owner, HttpMethod.GET, "/runs/" + absent + "/record", null).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test void persistedRunsRemainOwnedAndTerminalStopsAreImmutable() {
        String owner = cookie(PlatformRole.PROJECT_OWNER);
        String other = cookie(PlatformRole.PROJECT_OWNER);
        String admin = cookie(PlatformRole.PLATFORM_ADMIN);
        var me = http.exchange("/api/auth/me", HttpMethod.GET,
                new HttpEntity<>(null, headers(owner)), JsonNode.class);
        String username = me.getBody().path("username").asText();
        Long ownerId = users.findByUsername(username).orElseThrow().getId();
        SimulationRun run = new SimulationRun();
        run.setId(UUID.randomUUID()); run.setCreatedBy(ownerId);
        run.setStatus(SimulationStatus.STOPPED); run.setConfigHash("a".repeat(64));
        run.setRecipeJson("{\"head\":\"linear\",\"evaluationSplit\":\"validation\"}");
        run.setCreatedAt(Instant.now()); run.setEndedAt(Instant.now()); run.setRequestedRounds(3);
        run = runs.saveAndFlush(run);
        String path = "/runs/" + run.getId();
        for (String suffix : new String[]{"", "/record", "/stop"}) {
            assertThat(request(other, suffix.equals("/stop") ? HttpMethod.POST : HttpMethod.GET, path + suffix, "{}").getStatusCode())
                    .isEqualTo(HttpStatus.NOT_FOUND);
        }
        var detail = request(owner, HttpMethod.GET, path, null);
        assertThat(detail.getBody().path("completedRounds").isNull()).isTrue();
        assertThat(detail.getBody().path("recordAvailable").asBoolean()).isFalse();
        assertThat(detail.getBody().toString()).doesNotContain("workerPid", "createdBy", "recipeJson");
        assertThat(request(owner, HttpMethod.GET, path + "/record", null).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(request(owner, HttpMethod.POST, path + "/stop", "{}").getBody()).isEqualTo(detail.getBody());
        assertThat(request(admin, HttpMethod.GET, path, null).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(request(other, HttpMethod.GET, "/runs", null).getBody().path("totalElements").asInt()).isZero();
        assertThat(request(admin, HttpMethod.GET, "/runs?scope=platform", null).getBody().path("totalElements").asInt()).isPositive();
        runs.deleteById(run.getId());
    }

    private HttpHeaders headers(String cookie) {
        HttpHeaders headers = new HttpHeaders(); headers.set(HttpHeaders.COOKIE, cookie); return headers;
    }
}
