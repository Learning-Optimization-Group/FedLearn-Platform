package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * OP-4 deploy-config guards for the {@code ec2demo} profile, which runs behind the nginx reverse
 * proxy that terminates TLS. Cheap classpath-resource checks (no Spring context) that fail loudly if
 * either hardening is dropped from the profile.
 */
class Ec2demoForwardedHeadersConfigTest {

    private Properties ec2demoProps() throws Exception {
        Properties props = new Properties();
        try (InputStream in = getClass().getResourceAsStream("/application-ec2demo.properties")) {
            assertNotNull(in, "application-ec2demo.properties must be on the classpath");
            props.load(in);
        }
        return props;
    }

    @Test
    void ec2demoTrustsReverseProxyForwardedHeaders() throws Exception {
        assertEquals("native", ec2demoProps().getProperty("server.forward-headers-strategy"),
                "ec2demo is fronted by nginx (OP-4); it must set server.forward-headers-strategy=native so "
                        + "Tomcat's RemoteIpValve restores the real client IP for the SE-4 per-IP throttle, the "
                        + "audit trail, and request.isSecure() — not the 127.0.0.1 proxy loopback");
    }

    @Test
    void ec2demoDefersCookieSecureToTheEnvironmentRatherThanHardcodingItInsecure() throws Exception {
        String cookieSecure = ec2demoProps().getProperty("app.auth.cookie.secure");
        assertNotNull(cookieSecure, "app.auth.cookie.secure must be present");
        assertTrue(cookieSecure.contains("APP_AUTH_COOKIE_SECURE"),
                "ec2demo must not hard-code the cookie Secure flag; it must defer to "
                        + "${APP_AUTH_COOKIE_SECURE:...} so ec2-bootstrap.sh can enable it once a Let's Encrypt "
                        + "cert is live — otherwise HTTPS automation ships a non-Secure session cookie (got: "
                        + cookieSecure + ")");
    }
}
