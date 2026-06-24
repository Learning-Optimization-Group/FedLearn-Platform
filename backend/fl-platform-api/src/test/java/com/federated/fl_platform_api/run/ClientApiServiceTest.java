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
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;

    @InjectMocks ClientApiService service;

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
}
