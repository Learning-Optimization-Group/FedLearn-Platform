package com.federated.fl_platform_api.security;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.security.KeyFactory;
import java.security.MessageDigest;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;
import java.util.Date;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * SE-12: the issued per-client cert is verified INDEPENDENTLY (JDK CertificateFactory, not BouncyCastle):
 * signed by the CA, bound to the subject, marked as an mTLS client cert (clientAuth EKU, not a CA),
 * short-lived, with a standard PKCS#8 key and a sha256 CA fingerprint.
 */
class FlClientCertificateAuthorityTest {

    private FlClientCertificateAuthority ca;

    private FlClientCertificateAuthority enabledCa(long validitySeconds) {
        FlClientCertificateAuthority c = new FlClientCertificateAuthority();
        ReflectionTestUtils.setField(c, "enabled", true);
        ReflectionTestUtils.setField(c, "configuredCaCertPem", "");   // -> generate an ephemeral CA
        ReflectionTestUtils.setField(c, "configuredCaKeyPem", "");
        ReflectionTestUtils.setField(c, "validitySeconds", validitySeconds);
        ReflectionTestUtils.invokeMethod(c, "init");                   // run @PostConstruct
        return c;
    }

    private static X509Certificate parse(String pem) throws Exception {
        return (X509Certificate) CertificateFactory.getInstance("X.509")
                .generateCertificate(new ByteArrayInputStream(pem.getBytes(StandardCharsets.US_ASCII)));
    }

    @BeforeEach
    void setUp() {
        ca = enabledCa(3600);
    }

    @Test
    void issued_cert_is_signed_by_the_ca_and_bound_to_the_subject() throws Exception {
        UUID runId = UUID.randomUUID();
        FlClientCertificateAuthority.IssuedClientCert issued = ca.issueClientCert("user-abc", runId);

        X509Certificate cert = parse(issued.clientCertPem());
        X509Certificate caCert = parse(issued.caCertPem());

        cert.verify(caCert.getPublicKey()); // throws on an invalid signature -> proves the CA signed it
        assertThat(cert.getSubjectX500Principal().getName())
                .contains("CN=user-abc").contains("OU=run-" + runId);
        assertThat(cert.getExtendedKeyUsage()).contains("1.3.6.1.5.5.7.3.2"); // clientAuth
        assertThat(cert.getBasicConstraints()).isEqualTo(-1);                 // NOT a CA
        assertThat(cert.getKeyUsage()[0]).isTrue();                           // digitalSignature
    }

    @Test
    void issued_cert_has_the_configured_short_validity() throws Exception {
        FlClientCertificateAuthority.IssuedClientCert issued = ca.issueClientCert("u", UUID.randomUUID());
        X509Certificate cert = parse(issued.clientCertPem());
        long lifetimeSec = (cert.getNotAfter().getTime() - cert.getNotBefore().getTime()) / 1000;
        assertThat(lifetimeSec).isBetween(3600L, 3600L + 120L); // validity + the 60s skew backdate
        assertThat(cert.getNotAfter()).isAfter(new Date());
    }

    @Test
    void ca_fingerprint_is_the_sha256_of_the_ca_cert() throws Exception {
        FlClientCertificateAuthority.IssuedClientCert issued = ca.issueClientCert("u", UUID.randomUUID());
        byte[] d = MessageDigest.getInstance("SHA-256").digest(parse(issued.caCertPem()).getEncoded());
        StringBuilder hex = new StringBuilder();
        for (byte b : d) {
            hex.append(String.format("%02x", b));
        }
        assertThat(issued.caFingerprint()).isEqualTo(hex.toString()).isEqualTo(ca.caFingerprint());
    }

    @Test
    void distinct_subjects_get_distinct_certs_with_unique_serials() throws Exception {
        FlClientCertificateAuthority.IssuedClientCert a = ca.issueClientCert("alice", UUID.randomUUID());
        FlClientCertificateAuthority.IssuedClientCert b = ca.issueClientCert("bob", UUID.randomUUID());
        assertThat(a.clientCertPem()).isNotEqualTo(b.clientCertPem());
        assertThat(parse(a.clientCertPem()).getSerialNumber())
                .isNotEqualTo(parse(b.clientCertPem()).getSerialNumber());
    }

    @Test
    void the_private_key_pem_is_a_standard_pkcs8_rsa_key() throws Exception {
        FlClientCertificateAuthority.IssuedClientCert issued = ca.issueClientCert("u", UUID.randomUUID());
        assertThat(issued.clientKeyPem()).startsWith("-----BEGIN PRIVATE KEY-----");
        byte[] der = Base64.getDecoder().decode(
                issued.clientKeyPem().replaceAll("-----[A-Z ]+-----", "").replaceAll("\\s", ""));
        KeyFactory.getInstance("RSA").generatePrivate(new PKCS8EncodedKeySpec(der)); // throws if malformed
    }

    @Test
    void issuance_when_disabled_throws() {
        FlClientCertificateAuthority disabled = new FlClientCertificateAuthority();
        ReflectionTestUtils.setField(disabled, "enabled", false);
        assertThatThrownBy(() -> disabled.issueClientCert("u", UUID.randomUUID()))
                .isInstanceOf(IllegalStateException.class);
    }
}
