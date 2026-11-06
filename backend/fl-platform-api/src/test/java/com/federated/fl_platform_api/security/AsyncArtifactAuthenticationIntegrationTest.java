package com.federated.fl_platform_api.security;

import com.fedlearn.contract.v1.ArtifactRef;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.service.CustomUserDetailsService;
import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.core.Ordered;
import org.springframework.http.*;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.util.FileSystemUtils;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
@Import(AsyncArtifactAuthenticationIntegrationTest.ProbeConfiguration.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class AsyncArtifactAuthenticationIntegrationTest {
    private static final UUID ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final Path BUNDLES = temporaryBundles();
    @Autowired TestRestTemplate rest;
    @Autowired UserRepository users;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired ProjectMembershipRepository memberships;
    @Autowired OrganizationMembershipRepository organizations;
    @Autowired RunExecutionContractRepository contracts;
    @Autowired PlatformTransactionManager transactions;
    @Autowired JwtTokenProvider jwt;
    @Autowired CustomUserDetailsService details;
    @Autowired TokenRevocationService revocations;
    @Autowired CompletionProbe probe;

    record Fixture(User owner, User client, UUID runId, String sha, byte[] bytes) {
        String url() { return "/api/runs/" + runId + "/artifacts/" + sha; }
    }
    record Completion(int status, Exception failure) {}

    private static Path temporaryBundles() {
        try { return Files.createTempDirectory("fedlearn-async-artifacts-"); }
        catch (IOException error) { throw new ExceptionInInitializerError(error); }
    }
    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("app.model-bundle.dir", BUNDLES::toString);
    }
    @AfterAll
    static void cleanup() throws IOException { FileSystemUtils.deleteRecursively(BUNDLES); }

    private User user() {
        String id = UUID.randomUUID().toString();
        return users.saveAndFlush(new User("async-" + id, id + "@example.invalid", "test-only-unused-hash"));
    }
    private Fixture fixture() throws Exception {
        User owner = user();
        User client = user();
        Project project = new Project();
        project.setName("async-artifact-" + UUID.randomUUID());
        project.setUser(owner);
        project.setOrgId(ORG);
        project.setModelType("CNN");
        project.setModelName("net");
        project.setStatus("RUNNING");
        project = projects.saveAndFlush(project);
        memberships.saveAndFlush(new ProjectMembership(project, client, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, owner));
        Run run = new Run();
        run.setProjectId(project.getId());
        run.setRecipeKey("CNN");
        run.setStrategy("FedAvg");
        run.setStatus(RunStatus.RUNNING);
        run.setNumRounds(1);
        run.setMinClients(1);
        run.setClientsPerRound(1);
        run.setCreatedAt(Instant.now());
        run = runs.saveAndFlush(run);
        byte[] bytes = new byte[65536];
        for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) (i % 251);
        String sha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        Files.write(Files.createDirectories(BUNDLES.resolve(run.getId().toString())).resolve("loss.pte"), bytes);
        // Seed an immutable READY artifact fixture; contract validation is covered by its own corpus.
        byte[] contract = ExecutionContract.newBuilder().setRunId(run.getId().toString())
                .setProjectId(project.getId().toString()).setModelTraining(ModelTraining.newBuilder()
                    .addArtifacts(ArtifactVariant.newBuilder().addFiles(ArtifactRef.newBuilder()
                        .setRelativePath("loss.pte").setSha256(sha).setByteSize(bytes.length)))).build().toByteArray();
        UUID runId = run.getId();
        String contractId = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(contract));
        new TransactionTemplate(transactions).executeWithoutResult(ignored -> contracts.insertReady(runId, contract, contractId));
        return new Fixture(owner, client, runId, sha, bytes);
    }
    private String token(User user) {
        var principal = details.loadUserByUsername(user.getUsername());
        return jwt.generateToken(new UsernamePasswordAuthenticationToken(principal, null, principal.getAuthorities()));
    }
    private HttpHeaders authenticated(String token, boolean nativeClient) {
        HttpHeaders headers = new HttpHeaders();
        if (nativeClient) {
            headers.setBearerAuth(token);
            headers.set(JwtAuthenticationFilter.NATIVE_CLIENT_HEADER, "fedlearn-mobile");
        } else headers.set(HttpHeaders.COOKIE, "jwtToken=" + token);
        return headers;
    }
    private ResponseEntity<byte[]> download(Fixture fixture, HttpHeaders headers) {
        return rest.exchange(fixture.url(), HttpMethod.GET, new HttpEntity<>(headers), byte[].class);
    }
    private void denied(Fixture fixture, HttpHeaders headers, HttpStatus expected) {
        ResponseEntity<byte[]> response = download(fixture, headers);
        assertEquals(expected, response.getStatusCode());
        assertFalse(Arrays.equals(fixture.bytes(), response.getBody()), "Denied requests must not receive artifact bytes");
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void authorizedCookieAndNativeStreamsFinishWithoutLosingAuthentication(boolean nativeClient, boolean ranged) throws Exception {
        Fixture fixture = fixture();
        HttpHeaders headers = authenticated(token(nativeClient ? fixture.client() : fixture.owner()), nativeClient);
        if (ranged) headers.set(HttpHeaders.RANGE, "bytes=600-");
        String id = UUID.randomUUID().toString();
        headers.set(CompletionProbe.HEADER, id);
        CompletableFuture<Completion> completed = probe.expect(id);
        try {
            ResponseEntity<byte[]> response = download(fixture, headers);
            assertEquals(ranged ? HttpStatus.PARTIAL_CONTENT : HttpStatus.OK, response.getStatusCode());
            assertEquals('"' + fixture.sha() + '"', response.getHeaders().getETag());
            assertArrayEquals(ranged ? Arrays.copyOfRange(fixture.bytes(), 600, fixture.bytes().length) : fixture.bytes(), response.getBody());
            Completion finish = completed.get(5, TimeUnit.SECONDS);
            assertNull(finish.failure(), "Streaming completion must pass the real ASYNC security chain");
            assertEquals(ranged ? 206 : 200, finish.status());
            assertTrue(response.getHeaders().getOrEmpty(HttpHeaders.SET_COOKIE).stream().noneMatch(v -> v.contains("JSESSIONID")));
            denied(fixture, new HttpHeaders(), HttpStatus.UNAUTHORIZED);
        } finally { probe.pending.remove(id); }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void anonymousAndUnmarkedBearerRequestsRemainDenied(boolean unmarkedBearer) throws Exception {
        Fixture fixture = fixture();
        HttpHeaders headers = new HttpHeaders();
        if (unmarkedBearer) headers.setBearerAuth(token(fixture.owner()));
        denied(fixture, headers, HttpStatus.UNAUTHORIZED);
    }
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void revokedCookieAndNativeCredentialsRemainDenied(boolean nativeClient) throws Exception {
        Fixture fixture = fixture();
        String token = token(fixture.owner());
        revocations.revoke(jwt.getJti(token), jwt.getExpiration(token));
        denied(fixture, authenticated(token, nativeClient), HttpStatus.UNAUTHORIZED);
    }
    @Test
    void unrelatedUserCannotDownloadAnArtifact() throws Exception {
        Fixture fixture = fixture();
        denied(fixture, authenticated(token(user()), true), HttpStatus.FORBIDDEN);
    }
    @Test
    void clientMembershipCannotCrossTheCallersTenantBoundary() throws Exception {
        Fixture fixture = fixture();
        organizations.saveAndFlush(new OrganizationMembership(UUID.randomUUID(), fixture.client().getId(), OrgRole.MEMBER));
        denied(fixture, authenticated(token(fixture.client()), true), HttpStatus.FORBIDDEN);
    }

    /** Observe the dispatch after streaming; HTTP headers may already have reached the client. */
    static class CompletionProbe implements Filter {
        static final String HEADER = "X-Test-Async-Completion";
        final ConcurrentHashMap<String, CompletableFuture<Completion>> pending = new ConcurrentHashMap<>();
        CompletableFuture<Completion> expect(String id) {
            var future = new CompletableFuture<Completion>();
            pending.put(id, future);
            return future;
        }
        @Override
        public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
            Exception failure = null;
            try { chain.doFilter(request, response); }
            catch (IOException | ServletException | RuntimeException error) { failure = error; throw error; }
            finally {
                if (request.getDispatcherType() == DispatcherType.ASYNC) {
                    String id = ((HttpServletRequest) request).getHeader(HEADER);
                    var future = id == null ? null : pending.get(id);
                    if (future != null) future.complete(new Completion(((HttpServletResponse) response).getStatus(), failure));
                }
            }
        }
    }
    @TestConfiguration(proxyBeanMethods = false)
    static class ProbeConfiguration {
        @Bean CompletionProbe completionProbe() { return new CompletionProbe(); }
        @Bean FilterRegistrationBean<CompletionProbe> completionObserver(CompletionProbe probe) {
            var registration = new FilterRegistrationBean<>(probe);
            registration.setOrder(Ordered.HIGHEST_PRECEDENCE);
            registration.setAsyncSupported(true);
            registration.setDispatcherTypes(EnumSet.of(DispatcherType.REQUEST, DispatcherType.ASYNC));
            registration.addUrlPatterns("/api/runs/*");
            return registration;
        }
    }
}
