package com.federated.fl_platform_api;

import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.RunTokenRegistry;
import com.federated.fl_platform_api.service.ArtifactBlobStore;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.SpyBean;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/** Actual multipart callbacks must preserve run provenance and immutable retry semantics. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@ActiveProfiles("test")
class RunArtifactIngestIntegrationTest {
    @Autowired TestRestTemplate rest;
    @Autowired ProjectRepository projects;
    @Autowired RunRepository runs;
    @Autowired UserRepository users;
    @Autowired ModelArtifactRepository artifacts;
    @Autowired RunTokenRegistry tokens;
    @SpyBean ArtifactBlobStore blobStore;

    record ActiveRun(Project project, Run run, String token) {}

    private ActiveRun activeRun() {
        String suffix = UUID.randomUUID().toString();
        User owner = users.save(new User("artifact-" + suffix, suffix + "@example.com", "test-only-hash"));
        Project project = new Project();
        project.setName("artifact-" + suffix);
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

    private Map<String, String> metadata() {
        return new HashMap<>(Map.of("recipeKey", "CNN", "baseModelRef", "net", "licenseTag", "test-license",
                "evalCard", "{\"rounds\":1,\"source\":\"generated-control\"}"));
    }

    private byte[] weights() {
        return ("opaque-generated-control-" + UUID.randomUUID()).getBytes(StandardCharsets.UTF_8);
    }

    private ResponseEntity<JsonNode> upload(ActiveRun active, byte[] content, Map<String, String> metadata) {
        return upload(active, content, metadata, ArtifactKind.FULL_CHECKPOINT);
    }

    private ResponseEntity<JsonNode> upload(ActiveRun active, byte[] content, Map<String, String> metadata, ArtifactKind kind) {
        var form = new LinkedMultiValueMap<String, Object>();
        form.add("model", new ByteArrayResource(content) {
            @Override public String getFilename() { return "model.npz"; }
        });
        form.add("kind", kind.name());
        metadata.forEach(form::add);
        var headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        headers.set("X-Internal-Key", "test-internal-key");
        headers.set("X-Internal-Run-Token", active.token());
        return rest.exchange("/api/internal/projects/" + active.project().getId() + "/artifacts", HttpMethod.POST,
                new HttpEntity<>(form, headers), JsonNode.class);
    }

    @Test
    void currentRunStoresItsOwnProvenance() {
        ActiveRun active = activeRun();
        var response = upload(active, weights(), metadata());
        assertEquals(HttpStatus.OK, response.getStatusCode());
        var saved = artifacts.findById(UUID.fromString(response.getBody().path("id").asText())).orElseThrow();
        assertEquals(active.run().getId(), saved.getRunId());
        assertEquals(active.project().getId(), saved.getProjectId());
        assertEquals(active.project().getOrgId(), saved.getOrgId());
        assertEquals("CNN", saved.getRecipeKey());
    }

    @Test
    void anOldRunCannotPublishIntoTheNewActiveRun() {
        ActiveRun first = activeRun();
        ActiveRun current = newRun(first.project());
        assertEquals(HttpStatus.CONFLICT, upload(first, weights(), metadata()).getStatusCode());
        assertTrue(artifacts.findByProjectId(current.project().getId()).isEmpty());
        verify(blobStore, never()).put(any(byte[].class));
    }

    @Test
    void retiredAttemptCannotPublishButItsReplacementCan() {
        ActiveRun active = activeRun();
        String replacement = tokens.mint(active.project().getId(), active.run().getId());
        active.run().setInternalTokenHash(tokens.hash(replacement));
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.FORBIDDEN, upload(active, weights(), metadata()).getStatusCode());
        verify(blobStore, never()).put(any(byte[].class));
        assertEquals(HttpStatus.OK, upload(new ActiveRun(active.project(), active.run(), replacement), weights(), metadata()).getStatusCode());
        assertEquals(1, artifacts.findByRunId(active.run().getId()).size());
    }

    @Test
    void failedRunCannotCreateAnArtifact() {
        ActiveRun active = activeRun();
        active.run().setStatus(RunStatus.FAILED);
        runs.saveAndFlush(active.run());
        assertEquals(HttpStatus.CONFLICT, upload(active, weights(), metadata()).getStatusCode());
        assertTrue(artifacts.findByRunId(active.run().getId()).isEmpty());
        verify(blobStore, never()).put(any(byte[].class));
    }

    @Test
    void identicalReplayReturnsTheOriginalArtifactWithoutWritingAnotherBlob() {
        ActiveRun active = activeRun();
        byte[] content = weights();
        var first = upload(active, content, metadata());
        var equivalent = metadata();
        equivalent.put("evalCard", "{ \"source\": \"generated-control\", \"rounds\": 1 }");
        var second = upload(active, content, equivalent);
        assertEquals(HttpStatus.OK, first.getStatusCode());
        assertEquals(HttpStatus.OK, second.getStatusCode());
        assertEquals(first.getBody(), second.getBody());
        assertEquals(1, artifacts.findByRunId(active.run().getId()).size());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @Test
    void conflictingBytesCannotReplaceOrSilentlyReuseTheFirstArtifact() {
        ActiveRun active = activeRun();
        var first = upload(active, weights(), metadata());
        assertEquals(HttpStatus.OK, first.getStatusCode());
        assertEquals(HttpStatus.CONFLICT, upload(active, weights(), metadata()).getStatusCode());
        var saved = artifacts.findByRunId(active.run().getId());
        assertEquals(1, saved.size());
        assertEquals(first.getBody().path("sha256").asText(), saved.getFirst().getBlobSha256());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @ParameterizedTest
    @ValueSource(strings = {"baseModelRef", "licenseTag", "evalCard"})
    void conflictingMetadataCannotBeSilentlyAccepted(String field) {
        ActiveRun active = activeRun();
        byte[] content = weights();
        assertEquals(HttpStatus.OK, upload(active, content, metadata()).getStatusCode());
        var changed = metadata();
        changed.put(field, field.equals("evalCard") ? "{\"rounds\":2}" : "changed-value");
        assertEquals(HttpStatus.CONFLICT, upload(active, content, changed).getStatusCode());
        assertEquals(1, artifacts.findByRunId(active.run().getId()).size());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @Test
    void recipeCannotDisagreeWithTheRecordedRun() {
        ActiveRun active = activeRun();
        var changed = metadata();
        changed.put("recipeKey", "MLP");
        assertEquals(HttpStatus.BAD_REQUEST, upload(active, weights(), changed).getStatusCode());
        assertTrue(artifacts.findByRunId(active.run().getId()).isEmpty());
        verify(blobStore, never()).put(any(byte[].class));
    }

    @Test
    void omittedRecipeUsesTheRunSnapshotAfterTheProjectChanges() {
        ActiveRun active = activeRun();
        active.project().setModelType("MLP");
        projects.saveAndFlush(active.project());
        var omitted = metadata();
        omitted.remove("recipeKey");
        assertEquals(HttpStatus.OK, upload(active, weights(), omitted).getStatusCode());
        assertEquals("CNN", artifacts.findByRunId(active.run().getId()).getFirst().getRecipeKey());
    }

    @Test
    void concurrentRetriesPublishOneArtifactAndWriteOneBlob() throws Exception {
        ActiveRun active = activeRun();
        byte[] content = weights();
        var start = new CountDownLatch(1);
        try (var workers = Executors.newFixedThreadPool(4)) {
            var futures = new ArrayList<java.util.concurrent.Future<ResponseEntity<JsonNode>>>();
            for (int index = 0; index < 4; index++) {
                futures.add(workers.submit(() -> {
                    assertTrue(start.await(10, TimeUnit.SECONDS));
                    return upload(active, content, metadata());
                }));
            }
            start.countDown();
            String id = null;
            for (var future : futures) {
                var response = future.get(20, TimeUnit.SECONDS);
                assertEquals(HttpStatus.OK, response.getStatusCode());
                if (id == null) id = response.getBody().path("id").asText();
                assertEquals(id, response.getBody().path("id").asText());
            }
        }
        assertEquals(1, artifacts.findByRunId(active.run().getId()).size());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @Test
    void incompletePrivacyClaimIsStillRejectedBeforeBlobStorage() {
        ActiveRun active = activeRun();
        var claim = metadata();
        claim.put("evalCard", "{\"dp\":{\"enabled\":true}}");
        assertEquals(HttpStatus.BAD_REQUEST, upload(active, weights(), claim).getStatusCode());
        assertTrue(artifacts.findByRunId(active.run().getId()).isEmpty());
        verify(blobStore, never()).put(any(byte[].class));
    }

    @ParameterizedTest
    @ValueSource(strings = {" {}", " null", " trailing-content"})
    void appendedEvalCardContentCannotMasqueradeAsAnIdenticalReplay(String suffix) {
        ActiveRun active = activeRun();
        byte[] content = weights();
        assertEquals(HttpStatus.OK, upload(active, content, metadata()).getStatusCode());
        var changed = metadata();
        changed.put("evalCard", changed.get("evalCard") + suffix);
        assertEquals(HttpStatus.CONFLICT, upload(active, content, changed).getStatusCode());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @Test
    void legacyOpaqueCardsRetainExactReplayButRejectAChange() {
        ActiveRun active = activeRun();
        byte[] content = weights();
        var card = metadata();
        card.put("evalCard", "legacy-opaque-card");
        var first = upload(active, content, card);
        assertEquals(HttpStatus.OK, first.getStatusCode());
        var identical = upload(active, content, card);
        assertEquals(HttpStatus.OK, identical.getStatusCode());
        assertEquals(first.getBody(), identical.getBody());
        card.put("evalCard", "different-opaque-card");
        assertEquals(HttpStatus.CONFLICT, upload(active, content, card).getStatusCode());
        verify(blobStore, times(1)).put(any(byte[].class));
    }

    @ParameterizedTest
    @ValueSource(strings = {"LORA_ADAPTER", "BASE_REF"})
    void aCnnRunCannotPublishADifferentArtifactKind(String kind) {
        ActiveRun active = activeRun();
        assertEquals(HttpStatus.BAD_REQUEST, upload(active, weights(), metadata(), ArtifactKind.valueOf(kind)).getStatusCode());
        assertTrue(artifacts.findByRunId(active.run().getId()).isEmpty());
        verify(blobStore, never()).put(any(byte[].class));
    }

    @Test
    void aLoraRunOnlyPublishesAnAdapterForItsRecordedRecipe() {
        ActiveRun active = activeRun();
        active.run().setRecipeKey("LLM_LORA");
        active.run().setStrategy("FedLoRA");
        runs.saveAndFlush(active.run());
        var card = metadata();
        card.put("recipeKey", "LLM_LORA");
        assertEquals(HttpStatus.BAD_REQUEST, upload(active, weights(), card).getStatusCode());
        verify(blobStore, never()).put(any(byte[].class));
        assertEquals(HttpStatus.OK, upload(active, weights(), card, ArtifactKind.LORA_ADAPTER).getStatusCode());
        var artifact = artifacts.findByRunId(active.run().getId()).getFirst();
        assertEquals(ArtifactKind.LORA_ADAPTER, artifact.getKind());
        assertEquals("LLM_LORA", artifact.getRecipeKey());
    }
}
