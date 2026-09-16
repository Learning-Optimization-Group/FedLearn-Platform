package com.federated.fl_platform_api.security;

import jakarta.annotation.PostConstruct;

import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.asn1.x500.X500Name;
import org.bouncycastle.asn1.x509.BasicConstraints;
import org.bouncycastle.asn1.x509.ExtendedKeyUsage;
import org.bouncycastle.asn1.x509.Extension;
import org.bouncycastle.asn1.x509.KeyPurposeId;
import org.bouncycastle.asn1.x509.KeyUsage;
import org.bouncycastle.cert.X509CertificateHolder;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.cert.jcajce.JcaX509v3CertificateBuilder;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.openssl.PEMKeyPair;
import org.bouncycastle.openssl.PEMParser;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;
import org.bouncycastle.operator.ContentSigner;
import org.bouncycastle.operator.jcajce.JcaContentSignerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.StringReader;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.MessageDigest;
import java.security.PrivateKey;
import java.security.SecureRandom;
import java.security.Security;
import java.security.cert.X509Certificate;
import java.time.Instant;
import java.util.Base64;
import java.util.Date;
import java.util.UUID;

/**
 * SE-12: issues a short-lived, per-client X.509 mTLS certificate at enrollment, signed by a FL client CA.
 *
 * <p>A client that presents such a cert proves it holds an identity the platform ISSUED — one the
 * platform can rate-limit at issuance and REVOKE, turning the anonymous, only-TTL-bounded connection
 * token into an accountable identity. That is the Sybil-cost + revocation lever the marketplace threat
 * model prioritizes ({@code docs/security/marketplace-threat-model.md}).</p>
 *
 * <p>The FL gRPC server already VALIDATES client certs ({@code framework/.../server.py}:
 * {@code require_client_auth} + {@code root_certificates}); point its {@code FEDLEARN_GRPC_ROOT_CERT} at
 * this CA's cert to trust the issued certs. Feature-gated ({@code feature.fl-client-cert.enabled},
 * default OFF): flipping {@code require_client_auth} on AND wiring each client to present its cert is a
 * deployment/client step, so issuance ships dark first (mirrors how the SE-1 connection-token gate landed).</p>
 */
@Component
public class FlClientCertificateAuthority {

    private static final Logger log = LoggerFactory.getLogger(FlClientCertificateAuthority.class);
    private static final String BC = BouncyCastleProvider.PROVIDER_NAME;

    static {
        if (Security.getProvider(BC) == null) {
            Security.addProvider(new BouncyCastleProvider());
        }
    }

    /** The bundle handed to an enrolling client: its cert + private key (PEM), plus the CA cert +
     *  fingerprint so it can pin/verify the chain. */
    public record IssuedClientCert(String clientCertPem, String clientKeyPem, String caCertPem, String caFingerprint) {}

    @Value("${feature.fl-client-cert.enabled:false}")
    private boolean enabled;
    @Value("${app.fl.client-ca.cert-pem:}")
    private String configuredCaCertPem;
    @Value("${app.fl.client-ca.key-pem:}")
    private String configuredCaKeyPem;
    @Value("${app.fl.client-cert.validity-seconds:86400}")
    private long validitySeconds;

    private X509Certificate caCert;
    private PrivateKey caKey;
    private String caCertPem;
    private String caFingerprint;

    @PostConstruct
    void init() throws Exception {
        if (!enabled) {
            return;
        }
        if (!configuredCaCertPem.isBlank() && !configuredCaKeyPem.isBlank()) {
            caCert = parseCert(configuredCaCertPem);
            caKey = parseKey(configuredCaKeyPem);
            log.info("SE-12: FL client CA loaded from configuration.");
        } else {
            KeyPair caPair = generateRsa();
            caCert = selfSignedCa(caPair);
            caKey = caPair.getPrivate();
            log.warn("SE-12: FL client-cert issuance is enabled but no CA is configured — generated an "
                    + "EPHEMERAL in-memory CA (certs issued now won't verify after a restart). Set "
                    + "app.fl.client-ca.cert-pem + .key-pem (and the FL server's FEDLEARN_GRPC_ROOT_CERT) "
                    + "for a stable, verifiable CA in production.");
        }
        caCertPem = toPem("CERTIFICATE", caCert.getEncoded());
        caFingerprint = sha256Hex(caCert.getEncoded());
    }

    public boolean isEnabled() {
        return enabled;
    }

    public String caFingerprint() {
        return caFingerprint;
    }

    public String caCertPem() {
        return caCertPem;
    }

    /**
     * Issue a client cert bound to {@code subjectId} (the enrolling identity) for {@code runId}:
     * {@code CN=<subjectId>}, {@code OU=run-<runId>}, EKU=clientAuth, keyUsage=digitalSignature,
     * not-a-CA, {@code validity-seconds} lifetime, signed by the CA.
     */
    public IssuedClientCert issueClientCert(String subjectId, UUID runId) {
        if (!enabled || caKey == null || caCert == null) {
            throw new IllegalStateException("FL client-cert issuance is not enabled/initialized");
        }
        try {
            KeyPair clientPair = generateRsa();
            Instant now = Instant.now();
            X500Name issuer = new X500Name(caCert.getSubjectX500Principal().getName());
            X500Name subject = new X500Name("CN=" + subjectId + ",OU=run-" + runId);
            // 159-bit positive serial (never 0/negative), per RFC 5280 serial guidance.
            BigInteger serial = new BigInteger(159, new SecureRandom()).add(BigInteger.ONE);

            JcaX509v3CertificateBuilder b = new JcaX509v3CertificateBuilder(
                    issuer, serial,
                    Date.from(now.minusSeconds(60)),                 // small backdate for clock skew
                    Date.from(now.plusSeconds(validitySeconds)),
                    subject, clientPair.getPublic());
            b.addExtension(Extension.basicConstraints, true, new BasicConstraints(false));
            b.addExtension(Extension.keyUsage, true, new KeyUsage(KeyUsage.digitalSignature));
            b.addExtension(Extension.extendedKeyUsage, false, new ExtendedKeyUsage(KeyPurposeId.id_kp_clientAuth));

            ContentSigner signer = new JcaContentSignerBuilder("SHA256withRSA").setProvider(BC).build(caKey);
            X509Certificate cert = new JcaX509CertificateConverter().setProvider(BC).getCertificate(b.build(signer));

            return new IssuedClientCert(
                    toPem("CERTIFICATE", cert.getEncoded()),
                    toPem("PRIVATE KEY", clientPair.getPrivate().getEncoded()),
                    caCertPem, caFingerprint);
        } catch (Exception e) {
            throw new IllegalStateException("client cert issuance failed for " + subjectId, e);
        }
    }

    // ---- helpers ----

    private static KeyPair generateRsa() throws Exception {
        KeyPairGenerator g = KeyPairGenerator.getInstance("RSA");
        g.initialize(2048);
        return g.generateKeyPair();
    }

    private X509Certificate selfSignedCa(KeyPair caPair) throws Exception {
        Instant now = Instant.now();
        X500Name name = new X500Name("CN=FedLearn FL Client CA");
        JcaX509v3CertificateBuilder b = new JcaX509v3CertificateBuilder(
                name, BigInteger.valueOf(1L), Date.from(now.minusSeconds(60)),
                Date.from(now.plusSeconds(3650L * 86400L)), name, caPair.getPublic());
        b.addExtension(Extension.basicConstraints, true, new BasicConstraints(true));
        b.addExtension(Extension.keyUsage, true, new KeyUsage(KeyUsage.keyCertSign | KeyUsage.cRLSign));
        ContentSigner signer = new JcaContentSignerBuilder("SHA256withRSA").setProvider(BC).build(caPair.getPrivate());
        return new JcaX509CertificateConverter().setProvider(BC).getCertificate(b.build(signer));
    }

    private static X509Certificate parseCert(String pem) throws Exception {
        try (PEMParser p = new PEMParser(new StringReader(pem))) {
            return new JcaX509CertificateConverter().setProvider(BC)
                    .getCertificate((X509CertificateHolder) p.readObject());
        }
    }

    private static PrivateKey parseKey(String pem) throws Exception {
        try (PEMParser p = new PEMParser(new StringReader(pem))) {
            Object o = p.readObject();
            JcaPEMKeyConverter conv = new JcaPEMKeyConverter().setProvider(BC);
            if (o instanceof PEMKeyPair kp) {
                return conv.getKeyPair(kp).getPrivate();
            }
            if (o instanceof PrivateKeyInfo pki) {
                return conv.getPrivateKey(pki);
            }
            throw new IllegalArgumentException("unsupported CA key PEM (expected an RSA private key)");
        }
    }

    private static String toPem(String type, byte[] der) {
        String b64 = Base64.getMimeEncoder(64, "\n".getBytes(StandardCharsets.US_ASCII)).encodeToString(der);
        return "-----BEGIN " + type + "-----\n" + b64 + "\n-----END " + type + "-----\n";
    }

    private static String sha256Hex(byte[] der) throws Exception {
        byte[] d = MessageDigest.getInstance("SHA-256").digest(der);
        StringBuilder sb = new StringBuilder(64);
        for (byte x : d) {
            sb.append(Character.forDigit((x >> 4) & 0xF, 16)).append(Character.forDigit(x & 0xF, 16));
        }
        return sb.toString();
    }
}
