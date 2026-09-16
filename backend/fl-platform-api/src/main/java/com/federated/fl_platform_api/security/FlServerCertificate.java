package com.federated.fl_platform_api.security;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.HexFormat;
import java.util.Optional;

/**
 * The FL gRPC server's certificate, as clients need it to verify the server.
 *
 * <p>A deployment serves the FL boundary over TLS and self-signs the server keypair ({@code scripts/ec2-bootstrap.sh},
 * {@code deploy/TLS.md}), so a client can verify the server only against that certificate. The backend already
 * carries its path in {@code FEDLEARN_GRPC_SERVER_CERT} for the FL servers it spawns; this reads the same file once at
 * startup, so enrollment can hand it to clients instead of an operator copying it to every machine.
 *
 * <p>Read only when {@code app.fl.require-tls} is on: a plaintext deployment hands out nothing. A configured path that
 * is missing or is not a certificate fails startup, because every FL server this backend spawns would fail on the
 * same file.
 */
@Component
public class FlServerCertificate {

    private static final Logger log = LoggerFactory.getLogger(FlServerCertificate.class);

    private final boolean tlsRequired;
    private final String pem;
    private final String fingerprint;

    public FlServerCertificate(@Value("${app.fl.require-tls:false}") boolean tlsRequired,
                               @Value("${app.fl.grpc.server-cert-path:}") String certPath) {
        this.tlsRequired = tlsRequired;
        if (!tlsRequired || certPath == null || certPath.isBlank()) {
            if (tlsRequired) {
                log.warn("FL gRPC TLS is required but app.fl.grpc.server-cert-path is not set: clients get no server "
                        + "certificate and can verify the FL server only against their system roots.");
            }
            this.pem = null;
            this.fingerprint = null;
            return;
        }
        try {
            byte[] bytes = Files.readAllBytes(Path.of(certPath));
            X509Certificate cert = (X509Certificate) CertificateFactory.getInstance("X.509")
                    .generateCertificate(new ByteArrayInputStream(bytes));
            this.pem = new String(bytes, StandardCharsets.US_ASCII);
            this.fingerprint = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(cert.getEncoded()));
        } catch (IOException | GeneralSecurityException | ClassCastException e) {
            throw new IllegalStateException("Cannot load the FL gRPC server certificate from " + certPath
                    + " (app.fl.grpc.server-cert-path): " + e.getMessage(), e);
        }
    }

    public boolean tlsRequired() {
        return tlsRequired;
    }

    /** The certificate PEM to hand to clients; empty on a plaintext deployment or when none is configured. */
    public Optional<String> pem() {
        return Optional.ofNullable(pem);
    }

    /** Lowercase hex SHA-256 of the certificate's DER encoding; empty whenever {@link #pem()} is. */
    public Optional<String> fingerprint() {
        return Optional.ofNullable(fingerprint);
    }
}
