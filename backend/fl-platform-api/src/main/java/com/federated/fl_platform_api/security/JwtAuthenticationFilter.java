package com.federated.fl_platform_api.security;


import com.federated.fl_platform_api.service.CustomUserDetailsService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.DisabledException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(JwtAuthenticationFilter.class);

    /**
     * Native-client marker header (SE-9). Its presence with a non-blank value is what allows an
     * {@code Authorization: Bearer} token to authenticate — see {@link #isNativeClient}.
     */
    public static final String NATIVE_CLIENT_HEADER = "X-FedLearn-Client";

    private final JwtTokenProvider jwtTokenProvider;

    private final CustomUserDetailsService customUserDetailsService;

    private final TokenRevocationService tokenRevocationService;

    public JwtAuthenticationFilter(JwtTokenProvider jwtTokenProvider,
                                   CustomUserDetailsService customUserDetailsService,
                                   TokenRevocationService tokenRevocationService) {
        this.jwtTokenProvider = jwtTokenProvider;
        this.customUserDetailsService = customUserDetailsService;
        this.tokenRevocationService = tokenRevocationService;
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain
    )throws ServletException, IOException {

        String username = null;

        // SE-9 — Bearer acceptance is scoped to native clients; the browser is strictly cookie-only.
        //
        // The documented browser auth contract is cookies-only: the JWT lives in an HttpOnly,
        // SameSite jwtToken cookie that JS cannot read (defeats XSS token exfiltration), and the SPA
        // never sends an Authorization header. Native clients (mobile/desktop) cannot rely on the
        // HttpOnly cookie — they read the accessToken from the /auth/login response body (see
        // AuthController), stash it in secure platform storage (Keychain / EncryptedSharedPreferences),
        // and replay it as `Authorization: Bearer <jwt>`.
        //
        // To keep those two worlds from bleeding into each other we:
        //   1. ALWAYS honor a valid jwtToken cookie (the browser path — unchanged); and
        //   2. accept a Bearer header ONLY when the request also carries the native-client marker
        //      (NATIVE_CLIENT_HEADER). A browser-origin request presenting a Bearer header but no
        //      marker is treated as anonymous — the header is ignored, never authenticated from.
        // This is deliberately fail-closed: absent the explicit marker, Bearer does nothing. Native
        // clients MUST send NATIVE_CLIENT_HEADER for their Bearer token to be accepted.
        String jwt = readJwtCookie(request);
        if (jwt == null && isNativeClient(request)) {
            jwt = readBearerToken(request);
        }

        if (jwt == null) {
            filterChain.doFilter(request, response);
            return;
        }

        try {
            username = jwtTokenProvider.getUsernameFromToken(jwt);

            if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
                UserDetails userDetails = customUserDetailsService.loadUserByUsername(username);

                if (jwtTokenProvider.validateToken(jwt, userDetails)
                        && !tokenRevocationService.isRevoked(jwtTokenProvider.getJti(jwt))) {  // SE-8: honor logout
                    UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(
                            userDetails, null, userDetails.getAuthorities()
                    );
                    authToken.setDetails(
                            new WebAuthenticationDetailsSource().buildDetails(request)
                    );

                    SecurityContextHolder.getContext().setAuthentication(authToken);
                }
            }
        } catch (UsernameNotFoundException e) {
            // Token references a deleted/disabled user. Treat as anonymous; let
            // downstream Spring Security produce a 401.
            log.info("JWT references unknown user; rejecting as anonymous");
        } catch (DisabledException e) {
            // Account is no longer ACTIVE (suspended / pending / soft-deleted):
            // CustomUserDetailsService throws before any principal is built, so a
            // suspension takes effect on the very next request — the still-valid
            // JWT cannot keep an ended account alive until token expiry.
            log.info("JWT references a non-active user; rejecting as anonymous");
        } catch (RuntimeException e) {
            // Catches expired, malformed, or signature-mismatched JWTs. We never
            // log the token itself, only the exception class, to avoid disclosure.
            log.warn("JWT validation failed: {}", e.getClass().getSimpleName());
        }

        filterChain.doFilter(request, response);
    }

    /**
     * SE-9: a request is treated as a native client (mobile/desktop) — and therefore allowed to
     * authenticate via a Bearer header — only when it carries a non-blank {@link #NATIVE_CLIENT_HEADER}.
     * Browser SPAs never set this header, so their requests fall through to the cookie-only path.
     * The marker is an explicit intent signal, not a secret; it does not weaken any check — a marked
     * request still runs the full signature/expiry validation and jti revocation checks below.
     */
    private static boolean isNativeClient(HttpServletRequest request) {
        String marker = request.getHeader(NATIVE_CLIENT_HEADER);
        return marker != null && !marker.isBlank();
    }

    /** Reads the browser auth cookie. Returns {@code null} when no {@code jwtToken} cookie is present. */
    private static String readJwtCookie(HttpServletRequest request) {
        if (request.getCookies() != null) {
            for (jakarta.servlet.http.Cookie cookie : request.getCookies()) {
                if ("jwtToken".equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }
        return null;
    }

    /** Extracts the token from an {@code Authorization: Bearer <jwt>} header, or {@code null}. */
    private static String readBearerToken(HttpServletRequest request) {
        String authHeader = request.getHeader("Authorization");
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            return authHeader.substring(7);
        }
        return null;
    }
}
