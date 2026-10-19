package com.federated.fl_platform_api.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.federated.fl_platform_api.contract.ExecutionContractStore;
import com.federated.fl_platform_api.contract.PublicationOutcome;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.fedlearn.contract.v1.ExecutionContract;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End to end within the backend: a started run's published execution contract reaches the manifest as ProtoJSON
 * through the application's own JSON serialisation, beside the legacy fields.
 */
@SpringBootTest
@ActiveProfiles("test")
class RunManifestContractIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final Path GOLDEN = Path.of("..", "..", "framework", "tests", "fixtures", "execution_contract_v1",
            "golden_tinynet_fedavg.binpb");

    @Autowired RunService runService;
    @Autowired ExecutionContractStore store;
    @Autowired RunRepository runRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;
    @Autowired ObjectMapper applicationJson;

    private Run startedRun() {
        User owner = userRepository.save(new User("manifest-" + System.nanoTime(),
                "manifest-" + System.nanoTime() + "@example.com", passwordEncoder.encode("Password1!")));
        Project p = new Project();
        p.setName("manifest-" + System.nanoTime());
        p.setModelType("TINYNET_GOLDEN");
        p.setModelName("tinynet_golden");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(ProjectVisibility.PRIVATE);
        p = projectRepository.save(p);
        return runService.createForStart(p, "FedAvg", 3, 2, 2);
    }

    @Test
    void aPublishedContractIsServedAsProtoJsonBesideTheLegacyFields() throws Exception {
        Run run = startedRun();
        ExecutionContract contract = ExecutionContract.parseFrom(Files.readAllBytes(GOLDEN)).toBuilder()
                .setRunId(run.getId().toString())
                .setProjectId(run.getProjectId().toString())
                .build();
        assertThat(store.publish(run, contract)).isEqualTo(PublicationOutcome.PUBLISHED);

        JsonNode manifest = applicationJson.valueToTree(
                runService.toManifest(runRepository.findById(run.getId()).orElseThrow()));

        assertThat(manifest.path("contractState").asText()).isEqualTo("READY");
        assertThat(manifest.path("contractId").asText()).matches("[0-9a-f]{64}");
        assertThat(manifest.path("executionContract").path("runId").asText()).isEqualTo(run.getId().toString());
        assertThat(manifest.path("executionContract").path("round").path("timeoutMs").isTextual())
                .as("ProtoJSON renders uint64 as a string").isTrue();
        assertThat(manifest.path("recipeKey").asText()).isEqualTo("TINYNET_GOLDEN");
    }

    @Test
    void aRunWaitingForItsContractIsServedAsPending() {
        Run run = startedRun();
        JsonNode manifest = applicationJson.valueToTree(runService.toManifest(run));
        assertThat(manifest.path("contractState").asText()).isEqualTo("PENDING");
        assertThat(manifest.path("executionContract").isMissingNode() || manifest.path("executionContract").isNull())
                .isTrue();
    }
}
