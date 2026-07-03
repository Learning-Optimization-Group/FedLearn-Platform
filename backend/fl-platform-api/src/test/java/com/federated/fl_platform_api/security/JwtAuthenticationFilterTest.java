package com.federated.fl_platform_api.security;

import com.federated.fl_platform_api.service.CustomUserDetailsService;
import jakarta.servlet.http.Cookie;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.when;

/**
 * SE-9: Bearer acceptance is scoped to native clients; the browser is strictly cookie-only.
 *
 * These tests pin the token-acquisition contract of {@link JwtAuthenticationFilter}:
 *   - a valid cookie ALWAYS authenticates (the browser path);
 *   - an {@code Authorization: Bearer} header authenticates ONLY when the request also carries
 *     the native-client marker header ({@link JwtAuthenticationFilter#NATIVE_CLIENT_HEADER});
 *   - a Bearer header WITHOUT the marker (a browser-origin request) is ignored — it does not
 *     authenticate.
 * The downstream signature/expiry validation and jti revocation checks are left unchanged, and
 * one test asserts a revoked native token is still rejected.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT) // token-acquisition branch varies per request shape
class JwtAuthenticationFilterTest {

    private static final String TOKEN = "valid.jwt.token";
    private static final String JTI = "jti-1";

    @Mock
    private JwtTokenProvider jwtTokenProvider;

    @Mock
    private CustomUserDetailsService customUserDetailsService;

    @Mock
    private TokenRevocationService tokenRevocationService;

    private JwtAuthenticationFilter filter;
    private UserDetails userDetails;

    @BeforeEach
    void setUp() {
        SecurityContextHolder.clearContext();
        filter = new JwtAuthenticationFilter(jwtTokenProvider, customUserDetailsService, tokenRevocationService);

        userDetails = User.withUsername("alice")
                .password("password")
                .authorities(Collections.emptyList())
                .build();

        // Happy-path stubs: a request that reaches the validation branch with TOKEN authenticates as alice.
        // Whether these are exercised depends on how the token is acquired (cookie vs. marked Bearer), hence lenient.
        when(jwtTokenProvider.getUsernameFromToken(TOKEN)).thenReturn("alice");
        when(customUserDetailsService.loadUserByUsername("alice")).thenReturn(userDetails);
        when(jwtTokenProvider.validateToken(TOKEN, userDetails)).thenReturn(true);
        when(jwtTokenProvider.getJti(TOKEN)).thenReturn(JTI);
        when(tokenRevocationService.isRevoked(JTI)).thenReturn(false);
    }

    @AfterEach
    void tearDown() {
        SecurityContextHolder.clearContext();
    }

    private Authentication runFilter(MockHttpServletRequest request) throws Exception {
        MockFilterChain chain = new MockFilterChain();
        filter.doFilterInternal(request, new MockHttpServletResponse(), chain);
        // The filter must always continue the chain regardless of the auth outcome.
        assertNotNull(chain.getRequest(), "filter must always call the downstream chain");
        return SecurityContextHolder.getContext().getAuthentication();
    }

    @Test
    void browserBearerWithoutNativeMarker_isNotAuthenticated() throws Exception {
        // Browser-origin request: a Bearer header but no cookie and no native marker.
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer " + TOKEN);

        Authentication auth = runFilter(request);

        assertNull(auth, "a Bearer header without the native marker must be ignored for browser-origin requests");
    }

    @Test
    void nativeBearerWithMarker_isAuthenticated() throws Exception {
        // Native client (mobile/desktop): Bearer header + the native-client marker.
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer " + TOKEN);
        request.addHeader(JwtAuthenticationFilter.NATIVE_CLIENT_HEADER, "mobile");

        Authentication auth = runFilter(request);

        assertNotNull(auth, "a native request (Bearer + marker) must authenticate");
        assertEquals("alice", auth.getName());
    }

    @Test
    void validCookie_isAuthenticated() throws Exception {
        // Browser path: a valid jwtToken cookie always authenticates, no marker required.
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.setCookies(new Cookie("jwtToken", TOKEN));

        Authentication auth = runFilter(request);

        assertNotNull(auth, "a valid jwtToken cookie must always authenticate");
        assertEquals("alice", auth.getName());
    }

    @Test
    void nativeBearerWithMarker_butRevokedToken_isNotAuthenticated() throws Exception {
        // Revocation/jti check must still apply on the native Bearer path.
        when(tokenRevocationService.isRevoked(JTI)).thenReturn(true);

        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer " + TOKEN);
        request.addHeader(JwtAuthenticationFilter.NATIVE_CLIENT_HEADER, "desktop");

        Authentication auth = runFilter(request);

        assertNull(auth, "a revoked token must not authenticate even on the native Bearer path");
    }

    @Test
    void bearerWithBlankMarker_isNotAuthenticated() throws Exception {
        // A present-but-blank marker is not a native client; Bearer stays ignored (fail closed).
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("Authorization", "Bearer " + TOKEN);
        request.addHeader(JwtAuthenticationFilter.NATIVE_CLIENT_HEADER, "   ");

        Authentication auth = runFilter(request);

        assertNull(auth, "a blank native marker must not enable Bearer acceptance");
    }
}
