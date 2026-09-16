package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ActiveProfiles("test")
class AuditEventEntityTest {

    @Autowired
    AuditEventRepository repo;

    @Test
    void persists_and_reads_audit_event() {
        AuditEvent e = AuditEvent.builder()
                .action(AuditAction.USER_LOGIN_SUCCEEDED)
                .actorUserId(42L)
                .build();

        AuditEvent saved = repo.saveAndFlush(e);

        AuditEvent found = repo.findById(saved.getId()).orElseThrow();
        assertThat(found.getAction()).isEqualTo(AuditAction.USER_LOGIN_SUCCEEDED);
        assertThat(found.getOccurredAt()).isNotNull();
        assertThat(found.getActorUserId()).isEqualTo(42L);
    }
}
