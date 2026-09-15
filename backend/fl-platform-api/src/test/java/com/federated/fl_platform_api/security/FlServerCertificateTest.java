package com.federated.fl_platform_api.security;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The FL gRPC server's certificate, as clients need it. A deployment self-signs the server keypair
 * (scripts/ec2-bootstrap.sh), so a client can verify the server only if the platform hands it that certificate.
 */
class FlServerCertificateTest {

    @TempDir
    Path dir;

    /** A real self-signed certificate PEM and its fingerprint, from the client CA's generator. */
    private static FlClientCertificateAuthority someSelfSignedCa() throws Exception {
        FlClientCertificateAuthority ca = new FlClientCertificateAuthority();
        ReflectionTestUtils.setField(ca, "enabled", true);
        ReflectionTestUtils.setField(ca, "configuredCaCertPem", "");
        ReflectionTestUtils.setField(ca, "configuredCaKeyPem", "");
        ReflectionTestUtils.setField(ca, "validitySeconds", 3600L);
        ca.init();
        return ca;
    }

    @Test
    void loadsTheConfiguredCertificateAndItsSha256Fingerprint() throws Exception {
        FlClientCertificateAuthority ca = someSelfSignedCa();
        Path pem = Files.writeString(dir.resolve("server.crt"), ca.caCertPem());

        FlServerCertificate cert = new FlServerCertificate(true, pem.toString());

        assertThat(cert.tlsRequired()).isTrue();
        assertThat(cert.pem()).contains(ca.caCertPem());
        assertThat(cert.fingerprint()).contains(ca.caFingerprint());
    }

    @Test
    void noConfiguredPathMeansNoCertificateToHandOut() {
        FlServerCertificate cert = new FlServerCertificate(true, "");
        assertThat(cert.pem()).isEmpty();
        assertThat(cert.fingerprint()).isEmpty();
    }

    @Test
    void aPlaintextDeploymentHandsOutNothingAndDoesNotReadTheFile() throws Exception {
        Path pem = Files.writeString(dir.resolve("server.crt"), someSelfSignedCa().caCertPem());
        FlServerCertificate cert = new FlServerCertificate(false, pem.toString());
        assertThat(cert.tlsRequired()).isFalse();
        assertThat(cert.pem()).isEmpty();
        assertThatCode(() -> new FlServerCertificate(false, dir.resolve("absent.crt").toString()))
                .doesNotThrowAnyException();
    }

    @Test
    void aConfiguredPathThatIsMissingFailsLoudly() {
        assertThatThrownBy(() -> new FlServerCertificate(true, dir.resolve("absent.crt").toString()))
                .isInstanceOf(IllegalStateException.class).hasMessageContaining("absent.crt");
    }

    @Test
    void aFileThatIsNotACertificateFailsLoudly() throws Exception {
        Path junk = Files.writeString(dir.resolve("server.crt"), "not a certificate");
        assertThatThrownBy(() -> new FlServerCertificate(true, junk.toString()))
                .isInstanceOf(IllegalStateException.class);
    }
}
