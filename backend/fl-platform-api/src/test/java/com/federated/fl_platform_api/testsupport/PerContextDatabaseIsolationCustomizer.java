package com.federated.fl_platform_api.testsupport;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.test.context.ContextCustomizer;
import org.springframework.test.context.MergedContextConfiguration;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.utility.DockerImageName;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Redirects each {@code test}-profile {@code @SpringBootTest} context onto its own Postgres database
 * (all on ONE shared Testcontainers container) so Hibernate {@code create-drop} teardown on context
 * eviction can never drop the schema out from under a still-cached sibling context.
 *
 * <h2>The bug this closes</h2>
 * Every {@code @ActiveProfiles("test")} context runs {@code ddl-auto=create-drop}. Before this fix
 * they all pooled against the single shared {@code jdbc:tc:...///fedlearn_test} database. The
 * TestContext cache holds up to 32 contexts; the suite builds ~20 distinct ones (vanilla, the
 * {@code RANDOM_PORT} web variant, several {@code @MockBean}/{@code @TestPropertySource} forks) plus
 * the {@code dev}-profile {@code V*MigrationTest} contexts. When the cache evicts one shared
 * {@code fedlearn_test} context, its create-drop teardown issues {@code DROP TABLE} against the
 * shared schema — and the next test that reuses a still-cached sibling context fails with
 * {@code relation "users" does not exist}. The suite was green only because the eviction order
 * happened to work out; the next full-context test tips it over.
 *
 * <h2>The fix</h2>
 * This customizer runs for every context (registered as a {@code ContextCustomizerFactory} SPI in
 * {@code META-INF/spring.factories}, so no test class changes). For {@code test}-profile contexts it
 * creates a fresh database on a single shared container and overrides the datasource URL to point at
 * it. Each distinct cached context therefore owns a private schema; its create-drop teardown drops
 * only its own database and cannot affect any sibling. The {@code test} profile keeps create-drop
 * and Flyway stays disabled — only the physical database name changes, per context.
 *
 * <p>The {@code dev}-profile {@code V*MigrationTest} contexts (each already on its own
 * {@code jdbc:tc:...///fedlearn_vN} database with Flyway on and {@code validate}/{@code none}) are
 * left untouched.
 *
 * <p>This customizer is stateless with class-based {@code equals}/{@code hashCode}, so it does not
 * change the TestContext cache key: cache-equal test classes still share one context (hence one
 * isolated database); cache-distinct contexts each get their own.
 */
final class PerContextDatabaseIsolationCustomizer implements ContextCustomizer {

    static final PerContextDatabaseIsolationCustomizer INSTANCE = new PerContextDatabaseIsolationCustomizer();

    private static final Logger log = LoggerFactory.getLogger(PerContextDatabaseIsolationCustomizer.class);

    private PerContextDatabaseIsolationCustomizer() {
    }

    @Override
    public void customizeContext(ConfigurableApplicationContext context, MergedContextConfiguration mergedConfig) {
        ConfigurableEnvironment env = context.getEnvironment();

        // Only the shared create-drop `test`-profile contexts need isolating. The dev-profile
        // migration contexts already own a private jdbc:tc: database and must be left alone.
        List<String> activeProfiles = Arrays.asList(env.getActiveProfiles());
        if (!activeProfiles.contains("test")) {
            return;
        }
        // Respect any test that has already chosen an explicit, non-shared jdbc:tc: database.
        String currentUrl = env.getProperty("spring.datasource.url");
        if (currentUrl != null && currentUrl.startsWith("jdbc:tc:") && !currentUrl.contains("fedlearn_test")) {
            return;
        }

        SharedContainer.IsolatedDatabase db = SharedContainer.newDatabase();

        Map<String, Object> overrides = new LinkedHashMap<>();
        overrides.put("spring.datasource.url", db.jdbcUrl());
        overrides.put("spring.datasource.driver-class-name", "org.postgresql.Driver");
        overrides.put("spring.datasource.username", db.username());
        overrides.put("spring.datasource.password", db.password());
        env.getPropertySources().addFirst(new MapPropertySource("per-context-test-db-isolation", overrides));

        log.debug("Isolated test context {} onto database {}", mergedConfig.getTestClass().getSimpleName(),
                db.jdbcUrl());
    }

    // Class-based identity keeps the TestContext cache key stable (all instances are equal).
    @Override
    public boolean equals(Object obj) {
        return obj != null && getClass() == obj.getClass();
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }

    /**
     * A single Postgres container, started once per JVM, that hosts one database per isolated test
     * context. Reaped by the Testcontainers Ryuk sidecar at JVM exit.
     */
    private static final class SharedContainer {

        // Same image the retired shared jdbc:tc:postgresql:16.6-alpine URL resolved to, so no extra pull.
        private static final DockerImageName IMAGE = DockerImageName.parse("postgres:16.6-alpine");
        private static final AtomicInteger SEQ = new AtomicInteger();
        private static final PostgreSQLContainer<?> CONTAINER;

        static {
            CONTAINER = new PostgreSQLContainer<>(IMAGE)
                    .withDatabaseName("fedlearn_test_root")
                    .withUsername("test")
                    .withPassword("test");
            CONTAINER.start();
        }

        private SharedContainer() {
        }

        static IsolatedDatabase newDatabase() {
            String name = "fedlearn_test_" + SEQ.incrementAndGet();
            try (Connection c = DriverManager.getConnection(
                    CONTAINER.getJdbcUrl(), CONTAINER.getUsername(), CONTAINER.getPassword());
                 Statement s = c.createStatement()) {
                // Database identifiers are server-generated (fedlearn_test_<n>) — no injection surface.
                s.execute("CREATE DATABASE \"" + name + "\"");
            } catch (SQLException e) {
                throw new IllegalStateException("Failed to create isolated test database " + name, e);
            }
            String jdbcUrl = "jdbc:postgresql://" + CONTAINER.getHost() + ":"
                    + CONTAINER.getMappedPort(PostgreSQLContainer.POSTGRESQL_PORT) + "/" + name;
            return new IsolatedDatabase(jdbcUrl, CONTAINER.getUsername(), CONTAINER.getPassword());
        }

        record IsolatedDatabase(String jdbcUrl, String username, String password) {
        }
    }
}
