package com.federated.fl_platform_api.membership;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.JoinedVia;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectAccessRequest;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectAccessRequestRepository;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

// BEFORE_CLASS forces a fresh context (and a freshly recreated H2 schema)
// before this test class runs. Without it, the AFTER_EACH @DirtiesContext on
// the controller integration tests drops the shared in-memory schema and
// leaves any cached context for this class with no tables.
@SpringBootTest
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
@Transactional
class MembershipPersistenceTest {

    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired ProjectMembershipRepository membershipRepository;
    @Autowired ProjectAccessRequestRepository requestRepository;

    @Test
    void project_persistsVisibilityAndModelHubFields() {
        User owner = new User("alice", "alice@example.com", "hash");
        userRepository.save(owner);

        Project p = new Project();
        p.setName("test-vis-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        p.setVisibility(ProjectVisibility.PUBLIC);
        p.setModelPublished(true);
        p.setModelDescription("test description");
        p.setModelTags("vision,demo");
        p.setModelPublishedAt(Instant.parse("2026-05-12T00:00:00Z"));

        Project saved = projectRepository.saveAndFlush(p);
        Project reloaded = projectRepository.findById(saved.getId()).orElseThrow();

        assertEquals(ProjectVisibility.PUBLIC, reloaded.getVisibility());
        assertTrue(reloaded.isModelPublished());
        assertEquals("test description", reloaded.getModelDescription());
        assertEquals("vision,demo", reloaded.getModelTags());
        assertEquals(Instant.parse("2026-05-12T00:00:00Z"), reloaded.getModelPublishedAt());
    }

    @Test
    void membership_persistsAndQueriesByProjectAndUser() {
        User owner = userRepository.save(new User("bob", "bob@example.com", "hash"));
        User client = userRepository.save(new User("carol", "carol@example.com", "hash"));

        Project p = new Project();
        p.setName("test-mem-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        projectRepository.saveAndFlush(p);

        ProjectMembership m = new ProjectMembership(
            p, client, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, owner);
        membershipRepository.saveAndFlush(m);

        Optional<ProjectMembership> found =
            membershipRepository.findByIdProjectIdAndIdUserId(p.getId(), client.getId());
        assertTrue(found.isPresent());
        assertEquals(MembershipRole.CLIENT, found.get().getRole());
        assertEquals(JoinedVia.OWNER_ADD, found.get().getJoinedVia());
        assertNull(found.get().getPartitionId());

        assertEquals(-1, membershipRepository.maxPartitionIdForProject(p.getId()));

        found.get().setPartitionId(5);
        membershipRepository.saveAndFlush(found.get());
        assertEquals(5, membershipRepository.maxPartitionIdForProject(p.getId()));
    }

    @Test
    void accessRequest_uniquePerProjectUserPair() {
        User owner = userRepository.save(new User("dave", "dave@example.com", "hash"));
        User requester = userRepository.save(new User("eve", "eve@example.com", "hash"));

        Project p = new Project();
        p.setName("test-req-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("CREATED");
        p.setUser(owner);
        projectRepository.saveAndFlush(p);

        ProjectAccessRequest req = new ProjectAccessRequest(p, requester, "let me in");
        requestRepository.saveAndFlush(req);

        Optional<ProjectAccessRequest> fetched =
            requestRepository.findByProjectIdAndUserId(p.getId(), requester.getId());
        assertTrue(fetched.isPresent());
        assertEquals(AccessRequestStatus.PENDING, fetched.get().getStatus());

        // Approve it.
        fetched.get().setStatus(AccessRequestStatus.APPROVED);
        fetched.get().setDecidedAt(Instant.now());
        fetched.get().setDecidedBy(owner);
        requestRepository.saveAndFlush(fetched.get());

        assertEquals(1, requestRepository.findByUserId(requester.getId()).size());
    }
}
