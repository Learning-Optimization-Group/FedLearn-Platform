package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.FlClientCertificateAuthority;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RunServiceTest {

    @Mock RunRepository runRepository;
    @Mock RunEnrollmentRepository enrollmentRepository;
    @Mock ProjectRepository projectRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;
    @Mock ConnectionTokenService tokenService;
    @Mock FlClientCertificateAuthority clientCa;
    @Mock com.federated.fl_platform_api.security.FlServerCertificate serverCert;

    @InjectMocks RunService runService;

    @BeforeEach
    void injectValues() {
        ReflectionTestUtils.setField(runService, "grpcHost", "localhost");
    }

    private Project project(UUID id) {
        Project p = new Project();
        p.setId(id);
        p.setModelType("CNN");
        return p;
    }

    @Test
    void createForStart_buildsStartingRunWithSeedAndRecipe() {
        UUID pid = UUID.randomUUID();
        Project p = project(pid);
        when(runRepository.save(any(Run.class))).thenAnswer(inv -> {
            Run r = inv.getArgument(0);
            if (r.getId() == null) r.setId(UUID.randomUUID());
            return r;
        });

        Run run = runService.createForStart(p, "FedAvg", 5, 2, 4);

        assertEquals(RunStatus.STARTING, run.getStatus());
        assertEquals(4, run.getClientsPerRound());
        assertEquals("CNN", run.getRecipeKey());
        assertEquals(PartitioningMode.SHARDED, run.getPartitioningMode());
        assertNotNull(run.getSeed());
        assertNotNull(run.getCreatedAt());
    }

    @Test
    void markRunning_setsRunningHostPortAndStartedAt() {
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(runRepository.save(any(Run.class))).thenAnswer(i -> i.getArgument(0));

        runService.markRunning(rid, 50001);

        assertEquals(RunStatus.RUNNING, r.getStatus());
        assertEquals(50001, r.getServerPort());
        assertNotNull(r.getServerHost());
        assertNotNull(r.getStartedAt());
    }

    @Test
    void markFailed_setsFailedAndEndedAt() {
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(runRepository.save(any(Run.class))).thenAnswer(i -> i.getArgument(0));

        runService.markFailed(rid);

        assertEquals(RunStatus.FAILED, r.getStatus());
        assertNotNull(r.getEndedAt());
    }

    @Test
    void markFailed_isNoOpOnAnAlreadyCompletedRun() {
        // Terminal status is write-once. A fast run reaches COMPLETED (its /finished callback fired)
        // and the start-probe catch — or the reconciler sweep — then calls markFailed; the terminal
        // guard must keep it COMPLETED, never revert to FAILED (the FAILED-clobbers-COMPLETED race).
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.COMPLETED);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));

        runService.markFailed(rid);

        assertEquals(RunStatus.COMPLETED, r.getStatus(), "a completed run must not be reverted to FAILED");
        verify(runRepository, never()).save(any(Run.class));
    }

    @Test
    void markRunning_isNoOpOnAnAlreadyCompletedRun() {
        // Same race, the other eager writer: a fast run can reach COMPLETED before the start thread's
        // markRunning executes. markRunning must not revert a terminal run to RUNNING.
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.COMPLETED);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));

        runService.markRunning(rid, 50001);

        assertEquals(RunStatus.COMPLETED, r.getStatus(), "a completed run must not be reverted to RUNNING");
        verify(runRepository, never()).save(any(Run.class));
    }

    @Test
    void getStatus_runningRunExposesEndpoint() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING); r.setServerHost("localhost"); r.setServerPort(50002);
        Project p = project(pid);
        User u = new User(); u.setId(7L);

        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        var dto = runService.getStatus(rid);
        assertEquals("RUNNING", dto.getStatus());
        assertEquals("localhost:50002", dto.getGrpcEndpoint());
    }

    @Test
    void getStatus_pendingRunHidesEndpoint() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.STARTING);
        Project p = project(pid);
        User u = new User(); u.setId(7L);

        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        var dto = runService.getStatus(rid);
        assertEquals("STARTING", dto.getStatus());
        assertNull(dto.getGrpcEndpoint());
    }

    @Test
    void getStatus_nonParticipant_throwsAccessDenied() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING);

        // Project owned by user 2, caller is user 7 (stranger)
        Project p = project(pid);
        User owner = new User(); owner.setId(2L);
        p.setUser(owner);

        User stranger = new User(); stranger.setId(7L);

        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(stranger);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.empty());

        assertThrows(AccessDeniedException.class, () -> runService.getStatus(rid));
    }

    @Test
    void getStatus_memberRoleRejected() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING);

        // Project owned by user 2, caller is user 7 with MEMBER (not CLIENT) role
        Project p = project(pid);
        User owner = new User(); owner.setId(2L);
        p.setUser(owner);

        User member = new User(); member.setId(7L);

        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(member);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, member, MembershipRole.MEMBER)));

        assertThrows(AccessDeniedException.class, () -> runService.getStatus(rid));
    }

    @Test
    void getStatus_unknownRun_throwsNotFound() {
        UUID rid = UUID.randomUUID();
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> runService.getStatus(rid));
    }

    @Test
    void getManifest_returnsAllFields() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();

        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING);
        r.setRecipeKey("CNN");
        r.setStrategy("FedAvg");
        r.setNumRounds(10);
        r.setClientsPerRound(4);
        r.setPartitioningMode(PartitioningMode.SHARDED);
        r.setSeed(42L);

        Project p = project(pid);
        User u = new User(); u.setId(7L);

        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        var dto = runService.getManifest(rid);

        assertEquals("CNN", dto.getRecipeKey());
        assertEquals("FedAvg", dto.getStrategy());
        assertEquals(10, dto.getNumRounds());
        assertEquals(4, dto.getClientsPerRound());
        assertEquals("SHARDED", dto.getPartitioningMode());
        assertEquals(42L, dto.getSeed());
        assertEquals(rid, dto.getRunId());
        assertEquals(pid, dto.getProjectId());
    }

    // ─── enroll tests ──────────────────────────────────────────────────────────

    private Run runningRun(UUID rid, UUID pid, int k, PartitioningMode mode) {
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING); r.setServerHost("localhost"); r.setServerPort(50005);
        r.setClientsPerRound(k); r.setPartitioningMode(mode);
        r.setStrategy("FedAvg"); r.setNumRounds(5); r.setRecipeKey("CNN");
        return r;
    }

    @Test
    void enroll_firstClientGetsPartitionZero() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(7L);
        Run r = runningRun(rid, pid, 4, PartitioningMode.SHARDED);
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 7L)).thenReturn(java.util.Optional.empty());
        when(enrollmentRepository.maxPartitionIdForRun(rid)).thenReturn(-1);
        when(enrollmentRepository.save(any(RunEnrollment.class))).thenAnswer(i -> i.getArgument(0));
        when(tokenService.mint(any(), anyLong())).thenReturn(
                new ConnectionTokenService.Minted("tok", java.time.Instant.now().plusSeconds(120)));

        var dto = runService.enroll(rid);
        assertEquals(0, dto.getPartitionId());
        assertEquals("localhost:50005", dto.getGrpcEndpoint());
        assertEquals("tok", dto.getConnectionToken());
        assertEquals("SHARD", dto.getClientKind());
        assertNotNull(dto.getManifest());
        // SE-12: cert issuance is OFF by default -> the enrollment carries no client cert/key.
        assertNull(dto.getClientCertPem());
        assertNull(dto.getClientKeyPem());
    }

    @Test
    void enroll_issuesAClientCertBoundToTheUser_whenCertIssuanceEnabled() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(7L);
        Run r = runningRun(rid, pid, 4, PartitioningMode.SHARDED);
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 7L)).thenReturn(java.util.Optional.empty());
        when(enrollmentRepository.maxPartitionIdForRun(rid)).thenReturn(-1);
        when(enrollmentRepository.save(any(RunEnrollment.class))).thenAnswer(i -> i.getArgument(0));
        when(tokenService.mint(any(), anyLong())).thenReturn(
                new ConnectionTokenService.Minted("tok", java.time.Instant.now().plusSeconds(120)));
        when(clientCa.isEnabled()).thenReturn(true);
        when(clientCa.issueClientCert("7", rid)).thenReturn(new FlClientCertificateAuthority.IssuedClientCert(
                "CERT-PEM", "KEY-PEM", "CA-PEM", "ca-fp-sha256"));

        var dto = runService.enroll(rid);

        // the bundle is delivered with its CA's fingerprint, kept apart from caFingerprint, which describes the
        // FL server the client dials (null here: this run recorded none).
        assertEquals("CERT-PEM", dto.getClientCertPem());
        assertEquals("KEY-PEM", dto.getClientKeyPem());
        assertEquals("ca-fp-sha256", dto.getClientCaFingerprint());
        assertNull(dto.getCaFingerprint());
        // the cert is bound to THIS caller's id, not an attacker-supplied one.
        verify(clientCa).issueClientCert("7", rid);
    }

    @Test
    void enroll_isIdempotentForSameUser() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(7L);
        Run r = runningRun(rid, pid, 4, PartitioningMode.SHARDED);
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        RunEnrollment existing = new RunEnrollment(new RunEnrollmentId(rid, 7L), 2, ClientKind.SHARD, java.time.Instant.now());
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 7L)).thenReturn(java.util.Optional.of(existing));
        when(enrollmentRepository.save(any(RunEnrollment.class))).thenAnswer(i -> i.getArgument(0));
        when(tokenService.mint(any(), anyLong())).thenReturn(
                new ConnectionTokenService.Minted("tok2", java.time.Instant.now().plusSeconds(120)));

        var dto = runService.enroll(rid);
        assertEquals(2, dto.getPartitionId());            // reuses existing partition
        verify(enrollmentRepository, never()).maxPartitionIdForRun(rid);
    }

    @Test
    void enroll_shardedRejectsWhenFull() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(9L);
        Run r = runningRun(rid, pid, 2, PartitioningMode.SHARDED); // K=2, next would be 2 -> full
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 9L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 9L)).thenReturn(java.util.Optional.empty());
        when(enrollmentRepository.maxPartitionIdForRun(rid)).thenReturn(1); // next = 2 == K

        assertThrows(ProjectStateException.class, () -> runService.enroll(rid));
    }

    @Test
    void enroll_localModeIsUncapped() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(9L);
        Run r = runningRun(rid, pid, 1, PartitioningMode.LOCAL); // K=1 but LOCAL => no cap
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 9L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 9L)).thenReturn(java.util.Optional.empty());
        when(enrollmentRepository.maxPartitionIdForRun(rid)).thenReturn(5);
        when(enrollmentRepository.save(any(RunEnrollment.class))).thenAnswer(i -> i.getArgument(0));
        when(tokenService.mint(any(), anyLong())).thenReturn(
                new ConnectionTokenService.Minted("tok", java.time.Instant.now().plusSeconds(120)));

        var dto = runService.enroll(rid);
        assertEquals(6, dto.getPartitionId());
        assertEquals("LOCAL", dto.getClientKind());
    }

    @Test
    void enroll_rejectsWhenRunNotRunning() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Project p = project(pid); User u = new User(); u.setId(9L);
        Run r = runningRun(rid, pid, 4, PartitioningMode.SHARDED);
        r.setStatus(RunStatus.STARTING); r.setServerPort(null);
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 9L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        assertThrows(ProjectStateException.class, () -> runService.enroll(rid));
    }

    // helper
    private ProjectMembership membership(Project p, User u, MembershipRole role) {
        return new ProjectMembership(p, u, role, JoinedVia.OWNER_ADD, u);
    }

    // ─── Robust aggregation settings on the run record ──────────────────────────────────────────
    // A run must say which Byzantine-robust rule actually ran. "Robust" alone cannot distinguish median
    // from Bulyan, which lets two different experiments share one label (the bug class V22 closed for
    // the training arm).

    private Run savedRun(String strategy, RobustAggregationSettings robust) {
        when(runRepository.save(any(Run.class))).thenAnswer(inv -> inv.getArgument(0));
        return runService.createForStart(project(UUID.randomUUID()), strategy, 5, 20, 20, robust);
    }

    @Test
    void createForStart_persistsTheRobustSettingsGiven() {
        Run run = savedRun("Robust", new RobustAggregationSettings(RobustMethod.BULYAN, 0.2, null, null));
        assertEquals(RobustMethod.BULYAN, run.getRobustMethod());
        assertEquals(0.2, run.getRobustByzantineFraction());
        assertNull(run.getRobustTrimRatio());
        assertNull(run.getCenteredClipTau());
    }

    @Test
    void createForStart_robustWithoutSettings_recordsTheServerDefaultRule() {
        // No settings means fl_server.py falls back to median; the record says so rather than staying
        // blank, so the manifest names the rule that really ran.
        Run run = savedRun("Robust", null);
        assertEquals(RobustMethod.MEDIAN, run.getRobustMethod());
        assertNull(run.getRobustByzantineFraction());
    }

    @Test
    void createForStart_nonRobust_recordsNoRobustSettings() {
        Run run = savedRun("FedAvg", null);
        assertNull(run.getRobustMethod());
        assertNull(run.getRobustByzantineFraction());
        assertNull(run.getRobustTrimRatio());
        assertNull(run.getCenteredClipTau());
    }

    @Test
    void getManifest_namesTheRobustRuleAndItsSettings() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING);
        r.setRecipeKey("CNN");
        r.setStrategy("Robust");
        r.setNumRounds(10);
        r.setClientsPerRound(20);
        r.setPartitioningMode(PartitioningMode.SHARDED);
        r.setRobustMethod(RobustMethod.CENTERED_CLIP);
        r.setCenteredClipTau(1.5);

        Project p = project(pid);
        User u = new User(); u.setId(7L);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        var dto = runService.getManifest(rid);

        assertEquals("Robust", dto.getStrategy());
        assertEquals("CENTERED_CLIP", dto.getRobustMethod());
        assertEquals(1.5, dto.getCenteredClipTau());
        assertNull(dto.getRobustByzantineFraction());
        assertNull(dto.getRobustTrimRatio());
    }

    // ─── Secure aggregation on the run record ───────────────────────────────────────────────────
    // The phone decides from the manifest whether it can take part, so the flag is stored, not inferred.

    @Test
    void createForStart_recordsSecureAggregationAndItsThreshold() {
        when(runRepository.save(any(Run.class))).thenAnswer(inv -> inv.getArgument(0));
        Run run = runService.createForStart(project(UUID.randomUUID()), "DeComFL", 5, 3, 3, null, 2);
        assertTrue(run.isSecureAggregation());
        assertEquals(2, run.getSecureAggThreshold());
    }

    @Test
    void createForStart_withoutAThreshold_recordsSecureAggregationOff() {
        when(runRepository.save(any(Run.class))).thenAnswer(inv -> inv.getArgument(0));
        Run run = runService.createForStart(project(UUID.randomUUID()), "DeComFL", 5, 3, 3, null, null);
        assertFalse(run.isSecureAggregation());
        assertNull(run.getSecureAggThreshold());
    }

    @Test
    void getManifest_saysWhetherTheRunIsSecure() {
        UUID rid = UUID.randomUUID();
        UUID pid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid); r.setProjectId(pid);
        r.setStatus(RunStatus.RUNNING);
        r.setRecipeKey("CNN");
        r.setStrategy("DeComFL");
        r.setNumRounds(5);
        r.setClientsPerRound(3);
        r.setPartitioningMode(PartitioningMode.SHARDED);
        r.setSecureAggregation(true);
        r.setSecureAggThreshold(3);

        Project p = project(pid);
        User u = new User(); u.setId(7L);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));

        var dto = runService.getManifest(rid);

        assertTrue(dto.isSecureAggregation());
        assertEquals(3, dto.getSecureAggThreshold());
    }

    // ─── Server trust: the FL server's certificate at enrollment ───────────────────────────────
    // A deployment requires TLS on the FL boundary and self-signs the server certificate, so a client can verify the
    // server only if enrollment hands that certificate over. Nothing did, so a desktop or phone client could not dial
    // TLS without someone copying the file by hand.

    private Run arrangeEnrollment(UUID rid, UUID pid) {
        Project p = project(pid); User u = new User(); u.setId(7L);
        Run r = runningRun(rid, pid, 4, PartitioningMode.SHARDED);
        when(runRepository.lockById(rid)).thenReturn(java.util.Optional.of(r));
        when(projectRepository.findById(pid)).thenReturn(java.util.Optional.of(p));
        when(authz.currentUser()).thenReturn(u);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 7L))
                .thenReturn(java.util.Optional.of(membership(p, u, MembershipRole.CLIENT)));
        when(enrollmentRepository.findByIdRunIdAndIdUserId(rid, 7L)).thenReturn(java.util.Optional.empty());
        when(enrollmentRepository.maxPartitionIdForRun(rid)).thenReturn(-1);
        when(enrollmentRepository.save(any(RunEnrollment.class))).thenAnswer(i -> i.getArgument(0));
        when(tokenService.mint(any(), anyLong())).thenReturn(
                new ConnectionTokenService.Minted("tok", java.time.Instant.now().plusSeconds(120)));
        return r;
    }

    @Test
    void enroll_handsTheClientTheServerCertificate_whenTlsIsRequired() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        Run r = arrangeEnrollment(rid, pid);
        r.setGrpcCaFingerprint("srv-fp");
        when(serverCert.tlsRequired()).thenReturn(true);
        when(serverCert.pem()).thenReturn(java.util.Optional.of("SERVER-PEM"));

        var dto = runService.enroll(rid);

        assertTrue(dto.isGrpcTls());
        assertEquals("SERVER-PEM", dto.getGrpcServerCertPem());
        assertEquals("srv-fp", dto.getCaFingerprint());
    }

    @Test
    void enroll_onAPlaintextDeployment_handsOutNoCertificate() {
        UUID rid = UUID.randomUUID(); UUID pid = UUID.randomUUID();
        arrangeEnrollment(rid, pid);
        when(serverCert.tlsRequired()).thenReturn(false);

        var dto = runService.enroll(rid);

        assertFalse(dto.isGrpcTls());
        assertNull(dto.getGrpcServerCertPem());
    }

    @Test
    void markRunning_recordsTheServerCertificateFingerprint_whenTlsIsRequired() {
        UUID rid = UUID.randomUUID();
        Run r = new Run(); r.setId(rid); r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(serverCert.tlsRequired()).thenReturn(true);
        when(serverCert.fingerprint()).thenReturn(java.util.Optional.of("srv-fp"));

        runService.markRunning(rid, 50001);

        assertEquals("srv-fp", r.getGrpcCaFingerprint());
    }

    @Test
    void markRunning_onAPlaintextDeployment_recordsNoFingerprint() {
        UUID rid = UUID.randomUUID();
        Run r = new Run(); r.setId(rid); r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(serverCert.tlsRequired()).thenReturn(false);

        runService.markRunning(rid, 50001);

        assertNull(r.getGrpcCaFingerprint());
    }
}
