package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SE-2: gRPC TLS must be the fail-closed DEFAULT on the deployment profiles. When
 * {@code app.fl.require-tls=true}, {@code FlServerManager.configureChildEnv} sets
 * {@code FEDLEARN_REQUIRE_TLS=1} on the spawned FL server and the framework then refuses to bind
 * plaintext (the env mechanism is proven in {@code FlServerManagerCommandTest}). This guard
 * pins the profile DEFAULTS so a deployment cannot silently regress to plaintext: ec2demo and
 * production require TLS; dev/test/base stay plaintext so local runs and the FL integration tests
 * can still bind an insecure port.
 */
class FlServerTlsProfileDefaultTest {

    private static String requireTls(String profileResource) throws IOException {
        Properties props = new Properties();
        try (InputStream in = FlServerTlsProfileDefaultTest.class.getResourceAsStream(profileResource)) {
            assertNotNull(in, "missing profile resource " + profileResource);
            props.load(in);
        }
        return props.getProperty("app.fl.require-tls");
    }

    @Test
    void deploymentProfiles_defaultToFailClosedTls() throws IOException {
        assertEquals("true", requireTls("/application-ec2demo.properties"),
                "ec2demo must default to fail-closed gRPC TLS (SE-2)");
        assertEquals("true", requireTls("/application-production.properties"),
                "production must default to fail-closed gRPC TLS (SE-2)");
    }

    @Test
    void devAndBaseStayPlaintext_soLocalRunsAndTestsWork() throws IOException {
        // dev must not force TLS (may be unset -> null, or explicitly false).
        assertNotEquals("true", requireTls("/application-dev.properties"),
                "dev must stay plaintext for local runnability");
        // base default is ${APP_FL_REQUIRE_TLS:false} -> resolves false when the env var is unset;
        // Properties does not expand the placeholder, so accept the unexpanded form too.
        String base = requireTls("/application.properties");
        assertTrue(base == null || base.contains("false") || base.startsWith("${"),
                "base app.fl.require-tls must default to false, was: " + base);
    }
}
