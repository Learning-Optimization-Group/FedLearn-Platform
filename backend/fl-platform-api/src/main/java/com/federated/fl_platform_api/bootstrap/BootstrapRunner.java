package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.Organization;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.OrganizationRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.core.env.Environment;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Base64;
import java.util.UUID;

/**
 * One-shot startup task that seeds the first PLATFORM_ADMIN user and the
 * Platform organisation when the database has none.
 *
 * <p>Behaviour:</p>
 * <ul>
 *   <li>If {@code app.bootstrap.admin-email} is blank → logs and returns (no-op).</li>
 *   <li>If any user with {@code platformRole == "PLATFORM_ADMIN"} exists → returns.</li>
 *   <li>Otherwise: find-or-create the Platform org by slug, create the admin user
 *       (status=ACTIVE, emailVerified=true), add an OWNER membership, and emit
 *       two audit events ({@link AuditAction#BOOTSTRAP_ORG_CREATED} and
 *       {@link AuditAction#BOOTSTRAP_ADMIN_CREATED}) with actor=null (system).</li>
 * </ul>
 *
 * <p>Password resolution:</p>
 * <ul>
 *   <li>If {@code app.bootstrap.admin-password} is set → used verbatim.</li>
 *   <li>Else if the {@code dev} profile is active → a 24-char URL-safe Base64
 *       random password is generated and WARN-logged once. Dev convenience only.</li>
 *   <li>Else → fail fast with {@link IllegalStateException}.</li>
 * </ul>
 *
 * <p>The bean is {@code @Profile("!test")} so the unit test profile never runs it
 * (the dedicated {@code BootstrapRunnerTest} uses the {@code dev} profile).</p>
 */
@Component
@Profile("!test")
public class BootstrapRunner implements ApplicationRunner {

    private static final Logger LOG = LoggerFactory.getLogger(BootstrapRunner.class);

    private final BootstrapProps props;
    private final UserRepository users;
    private final OrganizationRepository orgs;
    private final OrganizationMembershipRepository memberships;
    private final AuditEventRepository audits;
    private final PasswordEncoder encoder;
    private final Environment env;

    @org.springframework.beans.factory.annotation.Value("${app.fl-server.grpc-host:localhost}")
    private String grpcHost;

    public BootstrapRunner(BootstrapProps props,
                           UserRepository users,
                           OrganizationRepository orgs,
                           OrganizationMembershipRepository memberships,
                           AuditEventRepository audits,
                           PasswordEncoder encoder,
                           Environment env) {
        this.props = props;
        this.users = users;
        this.orgs = orgs;
        this.memberships = memberships;
        this.audits = audits;
        this.encoder = encoder;
        this.env = env;
    }

    /**
     * OP-3: a deployed profile with a non-client-reachable grpc-host (localhost/127.0.0.1/0.0.0.0/::1)
     * hands FL clients an address that resolves to their OWN machine, so they silently fail to connect
     * to the FL server. Return a problem message in that case, else empty. Static + pure for testing.
     */
    static java.util.Optional<String> grpcHostMisconfig(java.util.Collection<String> activeProfiles,
                                                         String grpcHost) {
        boolean deployed = activeProfiles.contains("ec2demo") || activeProfiles.contains("production");
        if (!deployed) {
            return java.util.Optional.empty();
        }
        String h = grpcHost == null ? "" : grpcHost.trim().toLowerCase(java.util.Locale.ROOT);
        if (h.isEmpty() || h.equals("localhost") || h.equals("127.0.0.1")
                || h.equals("0.0.0.0") || h.equals("::1")) {
            return java.util.Optional.of(
                    "app.fl-server.grpc-host is '" + grpcHost + "' under deployed profile(s) "
                    + activeProfiles + " — FL clients would dial their own machine and silently fail. "
                    + "Set FL_SERVER_GRPC_HOST to the server's client-reachable address.");
        }
        return java.util.Optional.empty();
    }

    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        grpcHostMisconfig(java.util.Arrays.asList(env.getActiveProfiles()), grpcHost).ifPresent(msg -> {
            LOG.error("[bootstrap] FATAL CONFIG: {}", msg);
            throw new IllegalStateException(msg);   // OP-3: fail loud rather than silently mis-route clients
        });

        if (props.adminEmail() == null || props.adminEmail().isBlank()) {
            LOG.info("[bootstrap] no admin email configured; skipping");
            return;
        }

        if (users.existsByPlatformRole(PlatformRole.PLATFORM_ADMIN)) {
            LOG.info("[bootstrap] platform admin already present; skipping");
            return;
        }

        String orgName = (props.platformOrgName() == null || props.platformOrgName().isBlank())
                ? "Platform"
                : props.platformOrgName();
        String orgSlug = slug(orgName);
        Organization org = orgs.findBySlug(orgSlug)
                .orElseGet(() -> orgs.save(new Organization(UUID.randomUUID(), orgName, orgSlug)));

        audits.save(AuditEvent.builder()
                .action(AuditAction.BOOTSTRAP_ORG_CREATED)
                .orgId(org.getId())
                .targetType("ORG")
                .targetId(org.getId().toString())
                .build());

        boolean[] generatedHolder = { false };
        String password = resolvePassword(generatedHolder);
        String username = (props.adminUsername() == null || props.adminUsername().isBlank())
                ? props.adminEmail().split("@")[0]
                : props.adminUsername();

        // User.id is @GeneratedValue(IDENTITY) — never call setId().
        User admin = users.findByEmail(props.adminEmail()).orElseGet(User::new);
        admin.setUsername(username);
        admin.setEmail(props.adminEmail());
        admin.setPassword(encoder.encode(password));
        admin.setPlatformRole(PlatformRole.PLATFORM_ADMIN);
        admin.setStatus(UserStatus.ACTIVE);
        admin.setEmailVerified(true);
        admin = users.save(admin);

        memberships.save(new OrganizationMembership(org.getId(), admin.getId(), OrgRole.OWNER));

        audits.save(AuditEvent.builder()
                .action(AuditAction.BOOTSTRAP_ADMIN_CREATED)
                .orgId(org.getId())
                .targetType("USER")
                .targetId(admin.getId().toString())
                .build());

        LOG.info("[bootstrap] seeded PLATFORM_ADMIN user '{}' and org '{}' (slug={})",
                username, orgName, orgSlug);

        if (generatedHolder[0]) {
            LOG.warn("BOOTSTRAP ADMIN PASSWORD: {} — change this immediately", password);
        }
    }

    private String resolvePassword(boolean[] generatedHolder) {
        if (props.adminPassword() != null && !props.adminPassword().isBlank()) {
            return props.adminPassword();
        }
        if (!isDevProfile()) {
            throw new IllegalStateException(
                    "app.bootstrap.admin-password is required in non-dev profiles");
        }
        generatedHolder[0] = true;
        byte[] buf = new byte[18];
        new SecureRandom().nextBytes(buf);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(buf);
    }

    private boolean isDevProfile() {
        return Arrays.asList(env.getActiveProfiles()).contains("dev");
    }

    /** Lowercase, alphanumerics-with-single-dash slug, trimmed of leading/trailing dashes. */
    private static String slug(String name) {
        return name.toLowerCase().replaceAll("[^a-z0-9]+", "-").replaceAll("(^-|-$)", "");
    }
}
