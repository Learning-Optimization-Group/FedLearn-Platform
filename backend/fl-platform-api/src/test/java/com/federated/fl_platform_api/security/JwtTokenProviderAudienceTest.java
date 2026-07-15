package com.federated.fl_platform_api.security;

import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.io.Decoders;
import io.jsonwebtoken.security.Keys;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.test.util.ReflectionTestUtils;

import javax.crypto.SecretKey;
import java.util.Base64;
import java.util.Collections;
import java.util.Date;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SE-20 (secondary): cross-system token-confusion defense. The FL connection token
 * ({@link ConnectionTokenService#AUDIENCE} = {@code fedlearn-fl-server}) and the web session JWT are
 * both HMAC-signed; if {@code app.fl.token-secret} ever equals {@code app.jwt.secret} (a shared dev
 * secret), an FL connection token could otherwise be replayed as a web session cookie (web validation
 * only checked username+expiry, no audience). The web JWT is now minted with a distinct audience
 * ({@code fedlearn-web}) and verification requires it, so a foreign-audience (FL) or audience-less
 * (legacy) token signed with the same key is rejected.
 */
class JwtTokenProviderAudienceTest {

    // >= 32 bytes so hmacShaKeyFor accepts it for HS256; base64-encoded like the real app.jwt.secret.
    private static final String SECRET_B64 =
            Base64.getEncoder().encodeToString("se20-web-jwt-audience-test-secret-key-256bits-ok".getBytes());

    private JwtTokenProvider provider;
    private SecretKey key;

    @BeforeEach
    void setUp() {
        provider = new JwtTokenProvider();
        ReflectionTestUtils.setField(provider, "jwtSecretString", SECRET_B64);
        ReflectionTestUtils.setField(provider, "jwtExpirationInMs", 3_600_000L);
        provider.init();
        key = Keys.hmacShaKeyFor(Decoders.BASE64.decode(SECRET_B64));
    }

    private Authentication authFor(String username) {
        UserDetails u = User.withUsername(username).password("x").authorities(Collections.emptyList()).build();
        return new UsernamePasswordAuthenticationToken(u, null, Collections.emptyList());
    }

    @Test
    void webTokenCarriesTheWebAudience() {
        String token = provider.generateToken(authFor("alice"));
        var aud = Jwts.parser().verifyWith(key).build().parseSignedClaims(token).getPayload().getAudience();
        assertTrue(aud.contains(JwtTokenProvider.WEB_AUDIENCE),
                "web JWT must carry aud=" + JwtTokenProvider.WEB_AUDIENCE);
    }

    @Test
    void rejectsForeignAudienceToken_flConnectionTokenReplay() {
        // an FL connection token (aud=fedlearn-fl-server) signed with the SAME key (shared-secret worst case)
        String flLike = Jwts.builder().subject("alice")
                .audience().add(ConnectionTokenService.AUDIENCE).and()
                .expiration(new Date(System.currentTimeMillis() + 60_000))
                .signWith(key).compact();
        assertThrows(JwtException.class, () -> provider.getUsernameFromToken(flLike));
    }

    @Test
    void rejectsAudienceLessToken() {
        // a legacy web token minted before SE-20 (no audience) — rejected (one-time re-login on rollout)
        String noAud = Jwts.builder().subject("alice")
                .expiration(new Date(System.currentTimeMillis() + 60_000))
                .signWith(key).compact();
        assertThrows(JwtException.class, () -> provider.getUsernameFromToken(noAud));
    }
}
