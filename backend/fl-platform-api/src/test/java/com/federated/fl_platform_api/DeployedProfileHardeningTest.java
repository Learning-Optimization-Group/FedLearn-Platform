package com.federated.fl_platform_api;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the security posture DEFAULTS of the deployed profiles so a deployment cannot silently
 * regress. Two Fable-5 findings:
 *
 * <ul>
 *   <li><b>SE-21</b> — the {@code production} profile must set the auth cookie {@code Secure=true}
 *       and {@code SameSite=Strict} explicitly rather than inherit the base {@code Secure=false}
 *       (which exists so local {@code dev} and the plain-HTTP {@code ec2demo} keep working). dev and
 *       ec2demo intentionally stay {@code Secure=false} and must NOT be tightened here.</li>
 *   <li><b>SE-24</b> — actuator {@code /health} DETAILS (reconciler counts, exception class names)
 *       must be restricted to {@code PLATFORM_ADMIN} on the deployed profiles (production + ec2demo),
 *       not shown to any authenticated user.</li>
 * </ul>
 */
class DeployedProfileHardeningTest {

    private static Properties load(String profileResource) throws IOException {
        Properties props = new Properties();
        try (InputStream in = DeployedProfileHardeningTest.class.getResourceAsStream(profileResource)) {
            assertNotNull(in, "missing profile resource " + profileResource);
            props.load(in);
        }
        return props;
    }

    /** The effective default of a {@code ${VAR:default}} placeholder, or the literal value. */
    private static String effective(String raw) {
        if (raw == null) {
            return null;
        }
        if (raw.startsWith("${") && raw.endsWith("}") && raw.contains(":")) {
            return raw.substring(raw.indexOf(':') + 1, raw.length() - 1);
        }
        return raw;
    }

    // ---- SE-21: production auth cookie must be Secure + SameSite=Strict ----

    @Test
    void production_authCookieIsSecureAndStrict() throws IOException {
        Properties p = load("/application-production.properties");
        assertEquals("true", effective(p.getProperty("app.auth.cookie.secure")),
                "SE-21: production must default the auth cookie to Secure=true");
        assertEquals("Strict", effective(p.getProperty("app.auth.cookie.same-site")),
                "SE-21: production must default the auth cookie to SameSite=Strict");
    }

    @Test
    void devAndEc2demo_stayInsecureCookie_forPlainHttp() throws IOException {
        // These run over plain HTTP by design; a Secure cookie would be silently dropped.
        assertEquals("false", effective(load("/application-dev.properties").getProperty("app.auth.cookie.secure")),
                "dev must keep Secure=false for local HTTP");
        assertEquals("false", effective(load("/application-ec2demo.properties").getProperty("app.auth.cookie.secure")),
                "ec2demo must keep Secure=false for the plain-HTTP demo");
    }

    // ---- SE-24: actuator /health details restricted to admins on deployed profiles ----

    @Test
    void deployedProfiles_restrictHealthDetailsToAdmin() throws IOException {
        for (String profile : new String[]{"/application-production.properties", "/application-ec2demo.properties"}) {
            String roles = load(profile).getProperty("management.endpoint.health.roles");
            String showDetails = load(profile).getProperty("management.endpoint.health.show-details");
            boolean adminGated = "PLATFORM_ADMIN".equals(roles) || "never".equals(showDetails);
            assertTrue(adminGated,
                    "SE-24: " + profile + " must gate /health details to PLATFORM_ADMIN (roles) or hide them"
                            + " (show-details=never); got roles=" + roles + " show-details=" + showDetails);
        }
    }
}
