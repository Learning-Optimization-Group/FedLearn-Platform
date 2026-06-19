package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.dto.CreateProjectRequest;
import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.ModelInitializer;
import com.federated.fl_platform_api.service.ProjectService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.context.ActiveProfiles;

import java.util.Comparator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doNothing;

/**
 * Verifies the {@link Auditable @Auditable} aspect actually fires for a real
 * service mutation: creating a project writes exactly one {@code PROJECT_CREATED}
 * audit row. {@link ModelInitializer} is mocked so the test never spawns the
 * Python model-init process.
 */
@SpringBootTest
@ActiveProfiles("test")
class MutationAuditTest {

    @Autowired ProjectService projectService;
    @Autowired AuditEventRepository auditRepo;
    @Autowired UserRepository userRepository;
    @Autowired PasswordEncoder passwordEncoder;

    @MockBean ModelInitializer modelInitializer;

    @BeforeEach
    void seed() throws Exception {
        auditRepo.deleteAll();
        if (userRepository.findByUsername("mutator").isEmpty()) {
            User u = new User("mutator", "mutator@example.com",
                    passwordEncoder.encode("Password1!"));
            // Project creation is now gated on owner-or-admin (see ProjectService).
            u.setPlatformRole(PlatformRole.PROJECT_OWNER);
            userRepository.save(u);
        }
        // Neutralise the model-file initialisation (otherwise it forks Python).
        doNothing().when(modelInitializer)
                .initializeModelFile(any(), any(), any(), any(), anyInt());
    }

    @Test
    @WithMockUser(username = "mutator", roles = "PROJECT_OWNER")
    void createProject_writes_one_project_created_audit_row() throws Exception {
        long before = auditRepo.count();

        CreateProjectRequest req = new CreateProjectRequest();
        req.setName("audit-test-project");
        req.setModelType("CNN");
        req.setModelName("simple-cnn");
        req.setOptimizer("adam");
        req.setPretrainEpochs(0);

        projectService.createProject(req);

        assertThat(auditRepo.count()).isEqualTo(before + 1);

        AuditEvent newest = auditRepo.findAll().stream()
                .max(Comparator.comparing(AuditEvent::getOccurredAt))
                .orElseThrow();
        assertThat(newest.getAction()).isEqualTo(AuditAction.PROJECT_CREATED);
        assertThat(newest.getTargetType()).isEqualTo("PROJECT");
    }
}
