package com.federated.fl_platform_api.membership;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
@Transactional
class MembershipPersistenceTest {

    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;

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
}
