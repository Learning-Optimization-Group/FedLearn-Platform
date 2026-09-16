package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.stereotype.Component;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest
@ActiveProfiles("test")
@Import(AuditAspectTest.Cfg.class)
class AuditAspectTest {

    @Autowired AuditEventRepository repo;
    @Autowired AuditedBean bean;

    @BeforeEach void clear() { repo.deleteAll(); }

    @Test
    void writes_row_on_audited_method() {
        UUID userId = UUID.randomUUID();
        bean.doSomething(userId);
        assertThat(repo.findAll())
                .singleElement()
                .satisfies(e -> {
                    assertThat(e.getAction()).isEqualTo(AuditAction.USER_REGISTERED);
                    assertThat(e.getTargetId()).isEqualTo(userId.toString());
                });
    }

    @Test
    void rolls_back_with_caller_transaction() {
        assertThatThrownBy(() -> bean.failingTransactional(UUID.randomUUID()))
                .isInstanceOf(RuntimeException.class);
        assertThat(repo.findAll()).isEmpty();
    }

    @Test
    void merges_threadlocal_metadata() {
        UUID userId = UUID.randomUUID();
        bean.withContext(userId, "old=USER", "new=PLATFORM_ADMIN");
        assertThat(repo.findAll())
                .singleElement()
                .satisfies(e -> assertThat(e.getMetadata())
                        .contains("old=USER").contains("new=PLATFORM_ADMIN"));
    }

    @TestConfiguration
    static class Cfg { @Bean AuditedBean auditedBean() { return new AuditedBean(); } }

    @Component
    static class AuditedBean {
        @Auditable(action = AuditAction.USER_REGISTERED,
                   targetType = "USER", targetIdParam = "userId")
        public void doSomething(UUID userId) { /* no-op */ }

        @Transactional
        @Auditable(action = AuditAction.USER_PROFILE_UPDATED,
                   targetType = "USER", targetIdParam = "userId")
        public void failingTransactional(UUID userId) {
            throw new RuntimeException("boom");
        }

        @Auditable(action = AuditAction.USER_PLATFORM_ROLE_CHANGED,
                   targetType = "USER", targetIdParam = "userId")
        public void withContext(UUID userId, String oldVal, String newVal) {
            AuditContext.put("old_role", oldVal);
            AuditContext.put("new_role", newVal);
        }
    }
}
