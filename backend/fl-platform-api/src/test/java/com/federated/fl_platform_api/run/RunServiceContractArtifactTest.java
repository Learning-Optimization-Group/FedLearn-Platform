package com.federated.fl_platform_api.run;

import com.fedlearn.contract.v1.ArtifactRef;
import com.fedlearn.contract.v1.ArtifactVariant;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import com.federated.fl_platform_api.contract.ContractState;
import com.federated.fl_platform_api.contract.ContractView;
import com.federated.fl_platform_api.contract.ExecutionContractStore;
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
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Stage 3 slice A2: a phone downloads a run's artifacts by the SHA-256 its execution contract declares. The backend
 * serves a file only when the run's READY contract lists that hash, and only the file the contract names for it.
 */
@ExtendWith(MockitoExtension.class)
class RunServiceContractArtifactTest {

    @Mock RunRepository runRepository;
    @Mock RunEnrollmentRepository enrollmentRepository;
    @Mock ProjectRepository projectRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;
    @Mock ConnectionTokenService tokenService;
    @Mock ExecutionContractStore contractStore;
    @InjectMocks RunService runService;

    @TempDir Path modelsDir;

    private final UUID runId = UUID.randomUUID();
    private final UUID projectId = UUID.randomUUID();
    private final byte[] loss = "loss-graph-bytes".getBytes();

    @BeforeEach
    void setUp() throws Exception {
        ReflectionTestUtils.setField(runService, "objectMapper", new ObjectMapper());
        ReflectionTestUtils.setField(runService, "modelBundleDir", modelsDir.toString());
        ReflectionTestUtils.setField(runService, "bundleDeliveryEnabled", true);
        Files.write(Files.createDirectories(modelsDir.resolve(runId.toString())).resolve("loss.pte"), loss);
        Run run = new Run(); run.setId(runId); run.setProjectId(projectId); run.setStatus(RunStatus.RUNNING);
        Project p = new Project(); p.setId(projectId);
        User caller = new User(); caller.setId(7L);
        ProjectMembership m = mock(ProjectMembership.class);
        org.mockito.Mockito.lenient().when(m.getRole()).thenReturn(MembershipRole.CLIENT);
        org.mockito.Mockito.lenient().when(runRepository.findById(runId)).thenReturn(Optional.of(run));
        org.mockito.Mockito.lenient().when(projectRepository.findById(projectId)).thenReturn(Optional.of(p));
        org.mockito.Mockito.lenient().when(authz.currentUser()).thenReturn(caller);
        org.mockito.Mockito.lenient().when(membershipRepository.findByIdProjectIdAndIdUserId(projectId, 7L)).thenReturn(Optional.of(m));
    }

    private static String sha256(byte[] b) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));
    }

    private void contractListing(String relativePath, String sha, long size) {
        ExecutionContract c = ExecutionContract.newBuilder().setModelTraining(ModelTraining.newBuilder()
                .addArtifacts(ArtifactVariant.newBuilder().addFiles(ArtifactRef.newBuilder()
                        .setRelativePath(relativePath).setSha256(sha).setByteSize(size)))).build();
        when(contractStore.read(any())).thenReturn(
                new ContractView(ContractState.READY, c, c.toByteArray(), "id", null, null));
    }

    @Test
    void aFileTheContractListsIsServedByItsHash() throws Exception {
        String sha = sha256(loss);
        contractListing("loss.pte", sha, loss.length);

        RunService.ContractArtifact a = runService.getContractArtifact(runId, sha);

        assertEquals(modelsDir.resolve(runId.toString()).resolve("loss.pte"), a.file());
        assertEquals(sha, a.sha256());
        assertEquals(loss.length, a.byteSize());
    }

    @Test
    void aHashTheContractDoesNotListIsNotFound() throws Exception {
        contractListing("loss.pte", sha256(loss), loss.length);

        assertThrows(ResourceNotFoundException.class, () -> runService.getContractArtifact(runId, "a".repeat(64)));
    }

    @Test
    void aRunWithoutAReadyContractServesNothing() throws Exception {
        when(contractStore.read(any())).thenReturn(
                new ContractView(ContractState.PENDING, null, null, null, null, null));

        assertThrows(ResourceNotFoundException.class, () -> runService.getContractArtifact(runId, sha256(loss)));
    }

    @Test
    void aMalformedHashIsRejectedBeforeAnyLookup() {
        assertThrows(ResourceNotFoundException.class, () -> runService.getContractArtifact(runId, "../loss.pte"));
    }

    @Test
    void aContractPathThatEscapesTheBundleIsRefused() throws Exception {
        Files.write(modelsDir.resolve("outside.bin"), loss);
        String sha = sha256(loss);
        contractListing("../outside.bin", sha, loss.length);

        assertThrows(ResourceNotFoundException.class, () -> runService.getContractArtifact(runId, sha));
    }

    @Test
    void aStagedFileWhoseSizeDisagreesWithTheContractIsRefused() throws Exception {
        String sha = sha256(loss);
        contractListing("loss.pte", sha, loss.length + 1);

        assertThrows(ProjectStateException.class, () -> runService.getContractArtifact(runId, sha));
    }
}
