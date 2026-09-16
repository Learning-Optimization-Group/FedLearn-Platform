package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

/**
 * Regression guard for the shared-schema teardown coupling.
 *
 * <p>Every {@code @ActiveProfiles("test")} {@code @SpringBootTest} runs with Hibernate
 * {@code ddl-auto=create-drop}. Historically they all pooled against the single shared
 * {@code jdbc:tc:...///fedlearn_test} database, so when one Spring context was evicted from the
 * TestContext cache its create-drop teardown issued {@code DROP TABLE} against that shared schema —
 * pulling the tables out from under any still-cached sibling context and surfacing later as
 * {@code relation "users" does not exist}. The suite was only green by ordering luck.
 *
 * <p>This test forces the mechanism deterministically: it boots a second, independent
 * {@code test}-profile application context and closes it (firing its create-drop teardown), then
 * asserts THIS context's schema still exists. With per-context database isolation in place the
 * sibling teardown can only touch its own database, so our tables survive.
 */
@SpringBootTest
@ActiveProfiles("test")
class SharedSchemaIsolationProbeTest {

    @Autowired
    JdbcTemplate jdbc;

    @Test
    void testContextIsIsolatedOntoItsOwnDatabase() {
        // The whole fix hinges on this: a `test`-profile context must NOT sit on the shared
        // `fedlearn_test` database — it must be redirected to a private `fedlearn_test_<n>` one.
        String db = jdbc.queryForObject("SELECT current_database()", String.class);
        assertThat(db)
                .as("test contexts must be isolated onto a per-context database, not the shared fedlearn_test")
                .startsWith("fedlearn_test_")
                .isNotEqualTo("fedlearn_test");
    }

    @Test
    void siblingCreateDropTeardownMustNotDropThisContextsSchema() {
        // Sanity: this context's schema is present.
        assertThat(jdbc.queryForObject("SELECT to_regclass('public.users')", String.class))
                .as("precondition: this context's own schema is present")
                .isNotNull();

        // Boot a sibling full test-profile context (Hibernate create-drop, same application-test.properties)
        // and close it. Its create-drop teardown issues DROP TABLE on close.
        try (ConfigurableApplicationContext sibling = new SpringApplicationBuilder(FlPlatformApiApplication.class)
                .web(WebApplicationType.NONE)
                .profiles("test")
                .run()) {
            assertThat(sibling.isRunning()).isTrue();
        } // sibling.close() -> create-drop teardown fires here

        // This context's schema MUST survive the sibling's teardown. Pre-isolation, the sibling shared
        // fedlearn_test and dropped 'users' here -> this asserts null -> RED. With isolation -> GREEN.
        assertThatCode(() -> jdbc.queryForObject("SELECT count(*) FROM users", Integer.class))
                .as("sibling create-drop teardown must not drop this context's schema")
                .doesNotThrowAnyException();
        assertThat(jdbc.queryForObject("SELECT to_regclass('public.users')", String.class))
                .as("'users' table must still exist after a sibling context's create-drop teardown")
                .isNotNull();
    }
}
