package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.ClientProjectDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.service.ClientApiService;
import com.federated.fl_platform_api.service.RequirementsService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.access.AccessDeniedException;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ClientApiServiceTest {

    @Mock ProjectRepository projectRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock RunRepository runRepository;
    @Mock com.federated.fl_platform_api.service.RunService runService;
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;
    @Mock RequirementsService requirementsService;
    @Mock com.federated.fl_platform_api.service.ProjectStatusService projectStatusService;

    @InjectMocks ClientApiService service;

    @org.junit.jupiter.api.BeforeEach
    void stubDerivedStatus() {
        // BA-4: project status is derived from the active run; mock the deriver as the identity on
        // the stored status string so these DTO tests keep asserting the same values. Real
        // run->status derivation is covered by ProjectStatusServiceTest.
        org.mockito.Mockito.lenient().when(projectStatusService.currentStatus(org.mockito.ArgumentMatchers.any()))
            .thenAnswer(inv -> {
                String s = ((Project) inv.getArgument(0)).getStatus();
                return s == null ? com.federated.fl_platform_api.model.ProjectStatus.CREATED
                                 : com.federated.fl_platform_api.model.ProjectStatus.valueOf(s);
            });
    }

    private User user(long id) { User u = new User(); u.setId(id); return u; }

    private Project proj(UUID id, ProjectVisibility vis, User owner) {
        Project p = new Project();
        p.setId(id); p.setName("p-" + id); p.setModelType("CNN");
        p.setStatus("CREATED"); p.setVisibility(vis); p.setOrgId(UUID.randomUUID());
        p.setUser(owner);
        return p;
    }

    @Test
    void list_unionsMembershipsAndPublicDiscoverable_excludesPrivate() {
        User me = user(1L);
        when(authz.currentUser()).thenReturn(me);
        when(orgScope.isUnrestricted()).thenReturn(true);

        UUID mineId = UUID.randomUUID();
        Project mine = proj(mineId, ProjectVisibility.PRIVATE, me); // I own it
        when(projectRepository.findOwnedOrMemberOf(1L)).thenReturn(List.of(mine));

        UUID pubId = UUID.randomUUID();
        Project pub = proj(pubId, ProjectVisibility.PUBLIC, user(2L));
        UUID restrId = UUID.randomUUID();
        Project restr = proj(restrId, ProjectVisibility.RESTRICTED, user(2L));
        when(projectRepository.findDiscoverable(1L)).thenReturn(List.of(pub, restr));

        // mine is owned — isOwner=true so existsByIdProjectIdAndIdUserIdAndRole is not called
        // for pub/restr we only add PUBLIC
        List<ClientProjectDto> out = service.listForCurrentUser();

        // mine (owner) + public discoverable; RESTRICTED excluded from the client picker
        assertEquals(2, out.size());
        ClientProjectDto mineDto = out.stream().filter(d -> d.getProjectId().equals(mineId)).findFirst().orElseThrow();
        assertTrue(mineDto.isJoined());
        ClientProjectDto pubDto = out.stream().filter(d -> d.getProjectId().equals(pubId)).findFirst().orElseThrow();
        assertFalse(pubDto.isJoined());
        assertTrue(out.stream().noneMatch(d -> d.getProjectId().equals(restrId)));
    }

    @Test
    void join_publicCreatesClientMembership_idempotent() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project pub = proj(pid, ProjectVisibility.PUBLIC, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(pub));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 1L)).thenReturn(Optional.empty());

        ClientProjectDto dto = service.join(pid);

        assertTrue(dto.isJoined());
        verify(membershipRepository).save(any(ProjectMembership.class));
    }

    @Test
    void join_restrictedIsForbidden() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project restr = proj(pid, ProjectVisibility.RESTRICTED, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(restr));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 1L)).thenReturn(Optional.empty());

        assertThrows(AccessDeniedException.class, () -> service.join(pid));
    }

    @Test
    void join_privateIsNotFound() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project priv = proj(pid, ProjectVisibility.PRIVATE, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(priv));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.findByIdProjectIdAndIdUserId(pid, 1L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class, () -> service.join(pid));
    }

    @Test
    void getOne_publicNonMember_returnsJoinedFalse() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project pub = proj(pid, ProjectVisibility.PUBLIC, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(pub));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(pid, 1L, MembershipRole.CLIENT))
            .thenReturn(false);

        ClientProjectDto dto = service.getOne(pid);

        assertFalse(dto.isJoined());
        assertEquals(pid, dto.getProjectId());
    }

    @Test
    void getOne_privateNonMember_throwsNotFound() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project priv = proj(pid, ProjectVisibility.PRIVATE, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(priv));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(pid, 1L, MembershipRole.CLIENT))
            .thenReturn(false);

        assertThrows(ResourceNotFoundException.class, () -> service.getOne(pid));
    }

    @Test
    void getOne_clientMember_returnsJoinedTrue() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        Project priv = proj(pid, ProjectVisibility.PRIVATE, user(2L));
        when(projectRepository.findById(pid)).thenReturn(Optional.of(priv));
        when(orgScope.allows(any())).thenReturn(true);
        when(authz.currentUser()).thenReturn(me);
        when(membershipRepository.existsByIdProjectIdAndIdUserIdAndRole(pid, 1L, MembershipRole.CLIENT))
            .thenReturn(true);

        ClientProjectDto dto = service.getOne(pid);

        assertTrue(dto.isJoined());
        assertEquals(pid, dto.getProjectId());
    }

    @Test
    void getConnection_delegatesToEnrollAndReturnsLegacyShapePlusToken() {
        User me = user(1L);
        UUID pid = UUID.randomUUID();
        UUID rid = UUID.randomUUID();
        Project p = proj(pid, ProjectVisibility.PUBLIC, user(2L));
        p.setStatus("RUNNING");
        p.setActiveRunId(rid);
        when(projectRepository.findById(pid)).thenReturn(Optional.of(p));
        // authz.requireOrgScope is a mocked void — no stub needed; orgScope is not called directly.

        com.federated.fl_platform_api.dto.EnrollmentDto enr = new com.federated.fl_platform_api.dto.EnrollmentDto();
        enr.setGrpcEndpoint("localhost:50007");
        enr.setPartitionId(1);
        enr.setConnectionToken("tok");
        when(runService.enroll(rid)).thenReturn(enr);

        var dto = service.getConnection(pid);
        assertEquals("localhost:50007", dto.getServerAddress());
        assertEquals(1, dto.getPartitionId());
        assertEquals("tok", dto.getConnectionToken());
        assertEquals("CNN", dto.getModelType());
    }

    @Test
    void getConnection_includesRunStrategy_forClientStrategyDispatch() {
        UUID pid = UUID.randomUUID();
        UUID rid = UUID.randomUUID();
        Project p = proj(pid, ProjectVisibility.PUBLIC, user(2L));
        p.setStatus("RUNNING");
        p.setActiveRunId(rid);
        when(projectRepository.findById(pid)).thenReturn(Optional.of(p));

        com.federated.fl_platform_api.dto.EnrollmentDto enr = new com.federated.fl_platform_api.dto.EnrollmentDto();
        enr.setGrpcEndpoint("localhost:50007");
        enr.setPartitionId(0);
        enr.setConnectionToken("tok");
        when(runService.enroll(rid)).thenReturn(enr);

        com.federated.fl_platform_api.model.Run run = new com.federated.fl_platform_api.model.Run();
        run.setStrategy("DeComFL");
        when(runRepository.findById(rid)).thenReturn(Optional.of(run));

        var dto = service.getConnection(pid);
        // the desktop threads this into fl-runtime/client.py --strategy so a non-MLP DeComFL project
        // runs the DeComFL client path instead of silently defaulting to the FedAvg path.
        assertEquals("DeComFL", dto.getStrategy());
    }

    @Test
    void getConnection_noActiveRun_throwsProjectState() {
        UUID pid = UUID.randomUUID();
        Project p = proj(pid, ProjectVisibility.PUBLIC, user(2L));
        p.setActiveRunId(null);
        when(projectRepository.findById(pid)).thenReturn(Optional.of(p));
        // authz.requireOrgScope is a mocked void — no stub needed; orgScope is not called directly.

        assertThrows(com.federated.fl_platform_api.exception.ProjectStateException.class,
                () -> service.getConnection(pid));
    }

    @Test
    void toDto_populatesEffectiveRequirements() {
        User me = user(1L);
        when(authz.currentUser()).thenReturn(me);
        when(orgScope.isUnrestricted()).thenReturn(true);
        UUID pid = UUID.randomUUID();
        Project mine = proj(pid, ProjectVisibility.PUBLIC, me);
        when(projectRepository.findOwnedOrMemberOf(1L)).thenReturn(List.of(mine));
        when(projectRepository.findDiscoverable(1L)).thenReturn(List.of());
        com.federated.fl_platform_api.dto.DeviceRequirements req =
            new com.federated.fl_platform_api.dto.DeviceRequirements(8.0, null, null, null,
                Boolean.FALSE, null, null, null, null, null, null);
        when(requirementsService.effectiveFor(mine)).thenReturn(req);

        var out = service.listForCurrentUser();
        assertEquals(1, out.size());
        assertEquals(8.0, out.get(0).getRequirements().minRamGb());
        assertEquals(Boolean.FALSE, out.get(0).getRequirements().mobileSafe());
    }
}
