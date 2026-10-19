package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.RunService;
import com.fedlearn.contract.v1.ContractIssueCode;
import com.fedlearn.contract.v1.ExecutionContract;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;

import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The execution-contract publication lifecycle against a real database: PENDING until one decision is stored, READY
 * with immutable canonical bytes and a digest ID, UNAVAILABLE with a reason, LEGACY_ONLY for a run without an intent,
 * and exactly one winner when publication attempts race.
 */
@SpringBootTest
@ActiveProfiles("test")
class ExecutionContractStoreIntegrationTest {

    private static final UUID DEFAULT_ORG_ID = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired ExecutionContractStore store;
    @Autowired RunService runService;
    @Autowired RunRepository runRepository;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    private Run startedRun() {
        User owner = userRepository.save(new User("contract-" + System.nanoTime(),
                "contract-" + System.nanoTime() + "@example.com", passwordEncoder.encode("Password1!")));
        Project p = new Project();
        p.setName("contract-" + System.nanoTime());
        p.setModelType("TINYNET_GOLDEN");
        p.setModelName("tinynet_golden");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setOrgId(DEFAULT_ORG_ID);
        p.setVisibility(ProjectVisibility.PRIVATE);
        p = projectRepository.save(p);
        return runService.createForStart(p, "FedAvg", 3, 2, 2);
    }

    private Run legacyRun() {
        Run started = startedRun();
        Run legacy = new Run();
        legacy.setProjectId(started.getProjectId());
        legacy.setStrategy("FedAvg");
        legacy.setNumRounds(3);
        legacy.setMinClients(2);
        legacy.setClientsPerRound(2);
        legacy.setStatus(RunStatus.STARTING);
        legacy.setRecipeKey("TINYNET_GOLDEN");
        legacy.setCreatedAt(Instant.now());
        return runRepository.save(legacy);
    }

    private static ExecutionContract contractFor(Run run, int localEpochs) throws Exception {
        ExecutionContract golden = ExecutionContractCodec.parseBinary(ExecutionContractConformanceTest.goldenBytes());
        ExecutionContract.Builder b = golden.toBuilder()
                .setRunId(run.getId().toString())
                .setProjectId(run.getProjectId().toString());
        b.getModelTrainingBuilder().getLocalTrainingBuilder().setLocalEpochs(localEpochs);
        return b.build();
    }

    private static String sha256(byte[] bytes) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    @Test
    void aStartedRunIsPendingUntilDecidedAndARunWithoutAnIntentIsLegacyOnly() {
        assertThat(store.read(startedRun()).state()).isEqualTo(ContractState.PENDING);
        assertThat(store.read(legacyRun()).state()).isEqualTo(ContractState.LEGACY_ONLY);
    }

    @Test
    void aValidContractIsPublishedWithStableBytesAndADigestId() throws Exception {
        Run run = startedRun();
        ExecutionContract contract = contractFor(run, 5);

        assertThat(store.publish(run, contract)).isEqualTo(PublicationOutcome.PUBLISHED);

        ContractView first = store.read(runRepository.findById(run.getId()).orElseThrow());
        ContractView second = store.read(runRepository.findById(run.getId()).orElseThrow());
        assertThat(first.state()).isEqualTo(ContractState.READY);
        assertThat(first.contract()).isEqualTo(contract);
        assertThat(first.contractBytes()).isEqualTo(second.contractBytes());
        assertThat(first.contractId()).isEqualTo(sha256(first.contractBytes())).isEqualTo(second.contractId());
    }

    @Test
    void anInvalidContractIsRecordedUnavailableWithItsIssues() throws Exception {
        Run run = startedRun();
        ExecutionContract invalid = contractFor(run, 0);

        assertThat(store.publish(run, invalid)).isEqualTo(PublicationOutcome.MARKED_UNAVAILABLE);

        ContractView view = store.read(run);
        assertThat(view.state()).isEqualTo(ContractState.UNAVAILABLE);
        assertThat(view.unavailableReason()).isEqualTo(ContractUnavailableReason.INVALID_CONTRACT);
        assertThat(view.unavailableDetail()).contains(ContractIssueCode.ISSUE_OUT_OF_RANGE.name())
                .contains("modelTraining.localTraining.localEpochs");
        assertThat(view.contract()).isNull();
    }

    @Test
    void aContractForAnotherRunIsRefused() throws Exception {
        Run run = startedRun();
        ExecutionContract other = contractFor(startedRun(), 5);

        assertThat(store.publish(run, other)).isEqualTo(PublicationOutcome.MARKED_UNAVAILABLE);
        assertThat(store.read(run).unavailableDetail()).contains(ContractIssueCode.ISSUE_IDENTITY_MISMATCH.name());
    }

    @Test
    void aDecisionIsFinal() throws Exception {
        Run run = startedRun();
        ExecutionContract contract = contractFor(run, 5);
        store.publish(run, contract);

        assertThat(store.publish(run, contractFor(run, 6))).isEqualTo(PublicationOutcome.ALREADY_DECIDED);
        assertThat(store.markUnavailable(run, ContractUnavailableReason.STAGING_FAILED, "late failure"))
                .isEqualTo(PublicationOutcome.ALREADY_DECIDED);
        assertThat(store.read(run).contract()).isEqualTo(contract);
    }

    @Test
    void aRunCanBeMarkedUnavailableWithAReason() {
        Run run = startedRun();
        assertThat(store.markUnavailable(run, ContractUnavailableReason.STAGING_TIMED_OUT, "no bundle after 600 s"))
                .isEqualTo(PublicationOutcome.MARKED_UNAVAILABLE);
        ContractView view = store.read(run);
        assertThat(view.state()).isEqualTo(ContractState.UNAVAILABLE);
        assertThat(view.unavailableReason()).isEqualTo(ContractUnavailableReason.STAGING_TIMED_OUT);
    }

    @Test
    void aRunWithoutAnIntentNeverReceivesAContract() throws Exception {
        Run legacy = legacyRun();
        assertThatThrownBy(() -> store.publish(legacy, contractFor(legacy, 5)))
                .isInstanceOf(IllegalStateException.class);
        assertThat(store.read(legacy).state()).isEqualTo(ContractState.LEGACY_ONLY);
    }

    @Test
    void racingPublicationsProduceExactlyOneReadyContract() throws Exception {
        Run run = startedRun();
        int attempts = 8;
        ExecutorService pool = Executors.newFixedThreadPool(attempts);
        CountDownLatch go = new CountDownLatch(1);
        List<Future<PublicationOutcome>> futures = new ArrayList<>();
        try {
            for (int i = 0; i < attempts; i++) {
                ExecutionContract candidate = contractFor(run, i + 1);
                Callable<PublicationOutcome> attempt = () -> {
                    go.await();
                    return store.publish(run, candidate);
                };
                futures.add(pool.submit(attempt));
            }
            go.countDown();
            List<PublicationOutcome> results = new ArrayList<>();
            for (Future<PublicationOutcome> f : futures) {
                results.add(f.get());
            }
            assertThat(results).filteredOn(o -> o == PublicationOutcome.PUBLISHED).hasSize(1);
            assertThat(results).filteredOn(o -> o == PublicationOutcome.ALREADY_DECIDED).hasSize(attempts - 1);
        } finally {
            pool.shutdownNow();
        }
        ContractView view = store.read(run);
        assertThat(view.state()).isEqualTo(ContractState.READY);
        assertThat(view.contractId()).isEqualTo(sha256(view.contractBytes()));
    }
}
