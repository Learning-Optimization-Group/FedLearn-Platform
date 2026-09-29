package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.ModelBundleDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.service.RunService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.core.io.Resource;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** P2 verification: the model-bundle + whitelisted file endpoints, exercised against a real staged
 *  bundle with the participant-auth path mocked (mirrors RunServiceTest's conventions). */
@ExtendWith(MockitoExtension.class)
class RunServiceModelBundleTest {

    @Mock RunRepository runRepository;
    @Mock RunEnrollmentRepository enrollmentRepository;
    @Mock ProjectRepository projectRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;
    @Mock ConnectionTokenService tokenService;
    @Mock com.federated.fl_platform_api.contract.ExecutionContractStore contractStore;
    @Mock com.federated.fl_platform_api.service.ModelRecipeService modelRecipeService;
    @InjectMocks RunService runService;

    @TempDir Path modelsDir;

    @BeforeEach
    void inject() {
        ReflectionTestUtils.setField(runService, "objectMapper", new ObjectMapper());
        ReflectionTestUtils.setField(runService, "modelBundleDir", modelsDir.toString());
        ReflectionTestUtils.setField(runService, "bundleDeliveryEnabled", true);
        ReflectionTestUtils.setField(runService, "grpcHost", "localhost");
        org.mockito.Mockito.lenient().when(contractStore.read(org.mockito.ArgumentMatchers.any()))
                .thenReturn(new com.federated.fl_platform_api.contract.ContractView(
                        com.federated.fl_platform_api.contract.ContractState.LEGACY_ONLY, null, null, null, null,
                        null));
    }

    private static String sha256(byte[] b) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));
    }

    /** Stage a minimal bundle for {@code runId}; returns the expected loss/inputs sha for assertions. */
    private String[] stage(UUID runId) throws Exception {
        Path dir = Files.createDirectories(modelsDir.resolve(runId.toString()));
        byte[] loss = "loss-graph".getBytes(StandardCharsets.UTF_8);
        byte[] infer = "infer-graph".getBytes(StandardCharsets.UTF_8);
        byte[] inputs = new byte[8 * 4 * 4];   // [8,4] float32
        byte[] targets = new byte[8 * 8];      // [8] int64
        Files.write(dir.resolve("loss.pte"), loss);
        Files.write(dir.resolve("infer.pte"), infer);
        Files.write(dir.resolve("inputs.f32"), inputs);
        Files.write(dir.resolve("targets.i64"), targets);
        String lossSha = sha256(loss), inferSha = sha256(infer),
               inSha = sha256(inputs), tgtSha = sha256(targets);
        Files.writeString(dir.resolve("manifest.json"), """
            {
              "runId": "%s",
              "modelManifest": {
                "paramLayout": [{"name":"fc1.weight","shape":[5,4]},{"name":"fc1.bias","shape":[5]}],
                "totalParamCount": 43,
                "inferPtePath": "infer.pte",
                "inferSha256": "%s"
              },
              "lossPte": {"file":"loss.pte","sha256":"%s"},
              "dataset": {"inputsFile":"inputs.f32","inputsSha256":"%s","inputShape":[8,4],
                          "targetsFile":"targets.i64","targetsSha256":"%s","targetsShape":[8]}
            }""".formatted(runId, inferSha, lossSha, inSha, tgtSha));
        return new String[]{lossSha, inSha};
    }

    /** Mock the requireParticipantRun path so the caller is a CLIENT of the run's project. */
    private Run mockParticipant(UUID runId, UUID projectId) {
        Run run = new Run(); run.setId(runId); run.setProjectId(projectId);
        run.setStatus(RunStatus.RUNNING);
        Project p = new Project(); p.setId(projectId);
        User caller = new User(); caller.setId(7L);
        ProjectMembership m = mock(ProjectMembership.class);
        when(m.getRole()).thenReturn(MembershipRole.CLIENT);
        when(runRepository.findById(runId)).thenReturn(Optional.of(run));
        when(projectRepository.findById(projectId)).thenReturn(Optional.of(p));
        when(authz.currentUser()).thenReturn(caller);
        when(membershipRepository.findByIdProjectIdAndIdUserId(projectId, 7L)).thenReturn(Optional.of(m));
        return run;
    }

    /** A participant's run whose intent trains on {@code source}. */
    private void mockParticipant(UUID runId, UUID projectId, TrainingDataSource source) {
        Project p = new Project(); p.setId(projectId); p.setModelName("tinynet_golden");
        mockParticipant(runId, projectId).setIntent(RunIntent.capture(p, false, false, 900_000L, source));
    }

    @Test
    void getModelBundle_forARunOnParticipantsOwnData_advertisesNoFixtureData() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String[] shas = stage(rid);
        mockParticipant(rid, pid, TrainingDataSource.LOCAL_SNAPSHOT);

        ModelBundleDto b = runService.getModelBundle(rid);

        assertNull(b.inputsUrl());
        assertNull(b.inputsSha256());
        assertEquals(java.util.List.of(), b.inputShape());
        assertNull(b.targetsUrl());
        assertNull(b.targetsSha256());
        assertEquals(shas[0], b.lossSha256());
    }

    @Test
    void getModelFile_refusesFixtureDataForARunOnParticipantsOwnData() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stage(rid);
        mockParticipant(rid, pid, TrainingDataSource.LOCAL_SNAPSHOT);

        assertThrows(ResourceNotFoundException.class, () -> runService.getModelFile(rid, "inputs.f32"));
        assertThrows(ResourceNotFoundException.class, () -> runService.getModelFile(rid, "targets.i64"));
        assertNotNull(runService.getModelFile(rid, "loss.pte"));
    }

    @Test
    void getModelFile_servesFixtureDataForAFixtureRun() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String[] shas = stage(rid);
        mockParticipant(rid, pid, TrainingDataSource.FIXTURE);

        assertEquals("/api/runs/" + rid + "/files/inputs.f32", runService.getModelBundle(rid).inputsUrl());
        try (InputStream in = runService.getModelFile(rid, "inputs.f32").getInputStream()) {
            assertEquals(shas[1], sha256(in.readAllBytes()));
        }
    }

    @Test
    void getModelBundle_mapsManifestToDtoWithFileUrlsAndShas() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String[] shas = stage(rid);
        mockParticipant(rid, pid);

        ModelBundleDto b = runService.getModelBundle(rid);

        assertEquals(43, b.totalParamCount());
        assertEquals(2, b.paramLayout().size());
        assertEquals("fc1.weight", b.paramLayout().get(0).name());
        assertEquals(java.util.List.of(5, 4), b.paramLayout().get(0).shape());
        assertEquals("/api/runs/" + rid + "/files/loss.pte", b.lossPteUrl());
        assertEquals(shas[0], b.lossSha256());
        assertEquals("/api/runs/" + rid + "/files/infer.pte", b.inferPteUrl());
        assertEquals("/api/runs/" + rid + "/files/inputs.f32", b.inputsUrl());
        assertEquals(java.util.List.of(8, 4), b.inputShape());
        assertEquals(shas[1], b.inputsSha256());
    }

    @Test
    void getModelFile_streamsWhitelistedFileWithMatchingSha() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String[] shas = stage(rid);
        mockParticipant(rid, pid);

        Resource r = runService.getModelFile(rid, "loss.pte");
        try (InputStream in = r.getInputStream()) {
            assertEquals(shas[0], sha256(in.readAllBytes()));  // served bytes == manifest sha
        }
    }

    @Test
    void getModelFile_rejectsNonWhitelistedName() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stage(rid);
        mockParticipant(rid, pid);
        assertThrows(ResourceNotFoundException.class, () -> runService.getModelFile(rid, "manifest.json"));
        assertThrows(ResourceNotFoundException.class, () -> runService.getModelFile(rid, "../../etc/passwd"));
    }

    @Test
    void getModelBundle_missingBundle_throwsNotFound() {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        mockParticipant(rid, pid);  // participant, but nothing staged
        assertThrows(ResourceNotFoundException.class, () -> runService.getModelBundle(rid));
    }

    @Test
    void getModelBundle_featureDisabled_throws() {
        ReflectionTestUtils.setField(runService, "bundleDeliveryEnabled", false);
        assertThrows(ProjectStateException.class, () -> runService.getModelBundle(UUID.randomUUID()));
    }

    // ─── first-order (FedAvg) trainable bundle ────────────────────────────────────────────────────

    /** Extend a staged bundle with a first-order trainable.pte + the modelManifest trainable fields. */
    private String stageWithTrainable(UUID runId) throws Exception {
        stage(runId);
        Path dir = modelsDir.resolve(runId.toString());
        byte[] trainable = "trainable-graph".getBytes(StandardCharsets.UTF_8);
        Files.write(dir.resolve("trainable.pte"), trainable);
        String tSha = sha256(trainable);
        ObjectMapper om = new ObjectMapper();
        var m = (com.fasterxml.jackson.databind.node.ObjectNode)
                om.readTree(Files.readString(dir.resolve("manifest.json")));
        var mm = (com.fasterxml.jackson.databind.node.ObjectNode) m.get("modelManifest");
        mm.put("trainablePtePath", "trainable.pte");
        mm.put("trainableSha256", tSha);
        var names = mm.putArray("trainableParamNames");
        names.add("base.fc1.weight");
        names.add("base.fc1.bias");
        Files.writeString(dir.resolve("manifest.json"), om.writeValueAsString(m));
        return tSha;
    }

    /** A run complete enough for getManifest()/toManifest() (partitioningMode etc.), participant-mocked. */
    private void mockManifestRun(UUID runId, UUID projectId) {
        Run run = new Run();
        run.setId(runId);
        run.setProjectId(projectId);
        run.setStatus(RunStatus.RUNNING);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        Project p = new Project();
        p.setId(projectId);
        User caller = new User();
        caller.setId(7L);
        ProjectMembership m = mock(ProjectMembership.class);
        when(m.getRole()).thenReturn(MembershipRole.CLIENT);
        when(runRepository.findById(runId)).thenReturn(Optional.of(run));
        when(projectRepository.findById(projectId)).thenReturn(Optional.of(p));
        when(authz.currentUser()).thenReturn(caller);
        when(membershipRepository.findByIdProjectIdAndIdUserId(projectId, 7L)).thenReturn(Optional.of(m));
    }

    // Stage 3 D2: the device checks the trainable program against the exporter's probe before its first round.
    @Test
    void getModelBundle_carriesTheTrainableProgramsQualificationProbe() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stageWithTrainable(rid);
        Path manifest = modelsDir.resolve(rid.toString()).resolve("manifest.json");
        ObjectMapper om = new ObjectMapper();
        var m = (com.fasterxml.jackson.databind.node.ObjectNode) om.readTree(Files.readString(manifest));
        ((com.fasterxml.jackson.databind.node.ObjectNode) m.get("modelManifest")).putObject("trainableProbe")
                .put("rows", 8).put("width", 4).put("classes", 3).put("learningRate", 0.1)
                .put("lossStep1", 1.1053780317306519).put("lossStep2", 1.0776357650756836)
                .put("lossTolerance", 1e-4).put("maxProbeMs", 2000);
        Files.writeString(manifest, om.writeValueAsString(m));
        mockParticipant(rid, pid);

        ModelBundleDto.TrainableProbe probe = runService.getModelBundle(rid).trainableProbe();

        assertEquals(new ModelBundleDto.TrainableProbe(8, 4, 3, 0.1, 1.1053780317306519, 1.0776357650756836, 1e-4,
                2000), probe);
    }

    @Test
    void getModelBundle_withoutAProbe_carriesNone() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stageWithTrainable(rid);
        mockParticipant(rid, pid);
        assertNull(runService.getModelBundle(rid).trainableProbe());
    }

    @Test
    void getModelBundle_withTrainable_carriesTrainableUrlShaAndNames() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String tSha = stageWithTrainable(rid);
        mockParticipant(rid, pid);

        ModelBundleDto b = runService.getModelBundle(rid);

        assertEquals("/api/runs/" + rid + "/files/trainable.pte", b.trainablePteUrl());
        assertEquals(tSha, b.trainableSha256());
        assertEquals(java.util.List.of("base.fc1.weight", "base.fc1.bias"), b.trainableParamNames());
    }

    @Test
    void getModelBundle_withoutTrainable_trainableFieldsAreNullAndEmpty() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stage(rid);  // DeComFL-only bundle
        mockParticipant(rid, pid);

        ModelBundleDto b = runService.getModelBundle(rid);

        assertNull(b.trainablePteUrl());
        assertNull(b.trainableSha256());
        assertTrue(b.trainableParamNames().isEmpty());
    }

    @Test
    void getModelFile_servesWhitelistedTrainablePte() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        String tSha = stageWithTrainable(rid);
        mockParticipant(rid, pid);

        Resource r = runService.getModelFile(rid, "trainable.pte");
        try (InputStream in = r.getInputStream()) {
            assertEquals(tSha, sha256(in.readAllBytes()));  // served bytes == staged trainable sha
        }
    }

    @Test
    void getManifest_firstOrderSupported_trueOnlyWhenTrainableStaged() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stageWithTrainable(rid);
        mockManifestRun(rid, pid);
        assertTrue(runService.getManifest(rid).isFirstOrderSupported(),
                "a run with a staged trainable.pte advertises on-device first-order");

        UUID rid2 = UUID.randomUUID(), pid2 = UUID.randomUUID();
        stage(rid2);  // DeComFL-only — no trainable staged
        mockManifestRun(rid2, pid2);
        assertFalse(runService.getManifest(rid2).isFirstOrderSupported(),
                "a DeComFL-only run fail-closes first-order (phone stays on the ZO path)");
    }

    // Stage 3 B2: a phone imports its own data against the run's class list. The bundle carries it in the recipe's
    // order; the phone's snapshot then records labels-sha256 of that list, which must equal the contract's
    // labelSchemaId, so a substituted or reordered list cannot bind.
    @Test
    void getModelBundle_carriesTheRecipesClassNamesInOrder() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stage(rid);
        mockParticipant(rid, pid);
        projectRepository.findById(pid).orElseThrow().setModelType("TINYNET_GOLDEN");
        when(modelRecipeService.findByKey("TINYNET_GOLDEN")).thenReturn(Optional.of(
                new com.federated.fl_platform_api.dto.ModelRecipeDto("TINYNET_GOLDEN", "TinyNet", "vector",
                        java.util.List.of("c0", "c1", "c2"), java.util.List.of(), java.util.List.of(), null)));

        ModelBundleDto b = runService.getModelBundle(rid);

        assertEquals(java.util.List.of("c0", "c1", "c2"), b.classNames());
    }

    @Test
    void getModelBundle_hasNoClassNamesForARecipeTheCatalogDoesNotKnow() throws Exception {
        UUID rid = UUID.randomUUID(), pid = UUID.randomUUID();
        stage(rid);
        mockParticipant(rid, pid);
        projectRepository.findById(pid).orElseThrow().setModelType("RETIRED_RECIPE");
        when(modelRecipeService.findByKey("RETIRED_RECIPE")).thenReturn(Optional.empty());

        assertEquals(java.util.List.of(), runService.getModelBundle(rid).classNames());
    }
}

