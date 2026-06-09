package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Import;
import org.springframework.stereotype.Component;
import org.springframework.test.context.ActiveProfiles;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Proves {@link AuditAspect} serialises thread-local audit metadata into VALID
 * JSON (via Jackson), even when a value contains characters that hand-rolled
 * concatenation would corrupt: a double-quote, a backslash, and a newline.
 *
 * <p>This is the regression guard for the JSONB-correctness fix: under a JSONB
 * {@code audit_events.metadata} column, an invalid metadata string would fail
 * the audit insert and — because the aspect writes in the same transaction as
 * the audited mutation — roll the mutation back.
 */
@SpringBootTest
@ActiveProfiles("test")
@Import(AuditMetadataSerializationTest.Cfg.class)
class AuditMetadataSerializationTest {

    @Autowired AuditEventRepository repo;
    @Autowired MetaBean bean;
    @Autowired ObjectMapper objectMapper;

    @BeforeEach void clear() { repo.deleteAll(); }

    @Test
    void metadata_with_quote_backslash_newline_roundtrips_to_valid_json() throws Exception {
        // A project name a user could realistically pick that breaks naive JSON
        // string concatenation: a quote, a backslash, and an embedded newline.
        String hostile = "weird \"name\" with \\ and \nnewline";
        UUID userId = UUID.randomUUID();

        bean.withHostileMetadata(userId, hostile);

        AuditEvent e = repo.findAll().stream().findFirst().orElseThrow();
        String meta = e.getMetadata();
        assertThat(meta).isNotNull();

        // The whole metadata blob parses as valid JSON ...
        JsonNode root = objectMapper.readTree(meta);
        // ... and the hostile value round-trips byte-for-byte.
        assertThat(root.get("name").asText()).isEqualTo(hostile);
    }

    @TestConfiguration
    static class Cfg { @Bean MetaBean metaBean() { return new MetaBean(); } }

    @Component
    static class MetaBean {
        @Auditable(action = AuditAction.PROJECT_CREATED,
                   targetType = "PROJECT", targetIdParam = "userId")
        public void withHostileMetadata(UUID userId, String value) {
            AuditContext.put("name", value);
        }
    }
}
