package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.audit.Auditable;
import com.federated.fl_platform_api.dto.LoginRequest;
import com.federated.fl_platform_api.dto.RegisterRequest;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.AuditingAuthenticationFailureHandler;
import com.federated.fl_platform_api.security.AuditingAuthenticationSuccessHandler;
import com.federated.fl_platform_api.security.JwtAuthenticationFilter;
import com.federated.fl_platform_api.security.JwtTokenProvider;
import com.federated.fl_platform_api.security.LoginRateLimiter;
import com.federated.fl_platform_api.security.TokenRevocationService;
import com.federated.fl_platform_api.service.UserService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private static final Logger log = LoggerFactory.getLogger(AuthController.class);

    private final UserService userService;
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider tokenProvider;
    private final UserRepository userRepository;
    private final AuditingAuthenticationSuccessHandler successHandler;
    private final AuditingAuthenticationFailureHandler failureHandler;
    private final LoginRateLimiter loginRateLimiter;
    private final TokenRevocationService tokenRevocationService;

    @Value("${app.auth.cookie.secure:true}")
    private boolean cookieSecure;

    @Value("${app.auth.cookie.same-site:Strict}")
    private String cookieSameSite;

    // SE-8: the auth cookie must not outlive the JWT (a valid-looking cookie past the JWT's exp yields
    // silent 401s). Derive the cookie max-age from the JWT lifetime so the two can't drift.
    @Value("${app.jwt.expiration-ms}")
    private long jwtExpirationMs;

    @Autowired
    public AuthController(UserService userService, AuthenticationManager authenticationManager,
                          JwtTokenProvider tokenProvider, UserRepository userRepository,
                          AuditingAuthenticationSuccessHandler successHandler,
                          AuditingAuthenticationFailureHandler failureHandler,
                          LoginRateLimiter loginRateLimiter,
                          TokenRevocationService tokenRevocationService) {
        this.userService = userService;
        this.authenticationManager = authenticationManager;
        this.tokenProvider = tokenProvider;
        this.userRepository = userRepository;
        this.successHandler = successHandler;
        this.failureHandler = failureHandler;
        this.loginRateLimiter = loginRateLimiter;
        this.tokenRevocationService = tokenRevocationService;
    }

    @PostMapping("/register")
    @SuppressWarnings("null")
    // Caller is unauthenticated, so the aspect resolves actor=null. The generated user id
    // is unavailable as a method parameter, so we omit targetIdParam — the action enum
    // alone identifies the event. The aspect runs only after userService.registerUser
    // succeeds; failed registrations (duplicate username, validation error) write no audit row.
    @Auditable(action = AuditAction.USER_REGISTERED, targetType = "USER")
    public ResponseEntity<Map<String, Object>> registerUser(@Valid @RequestBody RegisterRequest registerRequest) {
        // UserAlreadyExistsException → 409, validation → 400, anything else → 500,
        // all centralised in GlobalExceptionHandler.
        User newUser = new User();
        newUser.setUsername(registerRequest.getUsername());
        newUser.setEmail(registerRequest.getEmail());
        newUser.setPassword(registerRequest.getPassword());

        User registeredUser = userService.registerUser(newUser);

        Map<String, Object> responseBody = Map.of(
                "message", "User registered successfully!",
                "userId", registeredUser.getId()
        );
        return ResponseEntity.status(HttpStatus.CREATED)
                .contentType(MediaType.APPLICATION_JSON)
                .body(responseBody);
    }

    @PostMapping("/login")
    @SuppressWarnings("null")
    public ResponseEntity<Map<String, Object>> authenticateUser(@Valid @RequestBody LoginRequest loginRequest,
                                                                HttpServletRequest http) {
        // AuthenticationException (bad credentials, disabled, locked, etc.) → 401 via
        // GlobalExceptionHandler. We catch it here only long enough to emit a
        // USER_LOGIN_FAILED audit row, then rethrow so the existing 401 path is unchanged.
        // SE-4: throttle brute-force. Block a locked-out username or source IP before even
        // attempting authentication; a valid login below clears the account's counter.
        String usernameKey = "u:" + loginRequest.getUsername();
        String ipKey = "ip:" + http.getRemoteAddr();
        if (loginRateLimiter.isLocked(usernameKey) || loginRateLimiter.isLocked(ipKey)) {
            // SE-4 (done-when #1): tell the caller how long to back off. Use the longer of the two
            // locked keys' remaining windows, rounded up to whole seconds (>= 1) per RFC 7231.
            long retryAfterSeconds = Stream.of(usernameKey, ipKey)
                    .map(loginRateLimiter::retryAfter)
                    .flatMap(Optional::stream)
                    .mapToLong(d -> Math.max(1L, (long) Math.ceil(d.toMillis() / 1000.0)))
                    .max()
                    .orElse(1L);
            log.warn("Login throttled for user '{}' from {} (retry after {}s)",
                    loginRequest.getUsername(), http.getRemoteAddr(), retryAfterSeconds);
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                    .header(HttpHeaders.RETRY_AFTER, Long.toString(retryAfterSeconds))
                    .body(Map.<String, Object>of("error", "Too many failed login attempts. Please try again later."));
        }

        Authentication authentication;
        try {
            authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            loginRequest.getUsername(),
                            loginRequest.getPassword()
                    )
            );
        } catch (AuthenticationException ex) {
            failureHandler.onFailure(loginRequest.getUsername(), http);
            loginRateLimiter.recordFailure(usernameKey);
            loginRateLimiter.recordFailure(ipKey);
            throw ex;
        }

        String authenticatedPrincipalName = authentication.getName();
        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = tokenProvider.generateToken(authentication);

        User appUser = userRepository.findByUsername(authenticatedPrincipalName)
                .orElseThrow(() -> {
                    // Should never happen — auth succeeded but the user row vanished mid-request.
                    log.error("Authenticated principal '{}' has no matching user row", authenticatedPrincipalName);
                    return ResourceNotFoundException.forEntity("User", authenticatedPrincipalName);
                });

        // Emit USER_LOGIN_SUCCEEDED audit row and update last_login_at BEFORE building
        // the response so a transient DB issue surfaces as a 500 rather than a half-committed login.
        successHandler.onSuccess(authenticatedPrincipalName, http);
        loginRateLimiter.reset(usernameKey); // a good password ends the throttle for this account

        ResponseCookie jwtCookie = ResponseCookie.from("jwtToken", jwt)
                .httpOnly(true)
                .secure(cookieSecure)
                .path("/")
                .maxAge(jwtExpirationMs / 1000)   // SE-8: cookie expires with the JWT
                .sameSite(cookieSameSite)
                .build();

        // The JWT is set as an HttpOnly cookie (defeats XSS exfiltration) for browser SPAs,
        // which ignore the body and rely on the cookie + GET /api/auth/me for session checks.
        // SE-8 (done-when #3): the browser must NOT receive a JS-readable token in the body.
        // Only native clients (mobile/desktop) — which cannot read the HttpOnly Set-Cookie and
        // replay the JWT as a Bearer from secure platform storage — get accessToken. They
        // self-identify with the X-FedLearn-Client marker, the same signal SE-9 gates Bearer
        // acceptance on, so a browser login response carries identity only.
        String clientMarker = http.getHeader(JwtAuthenticationFilter.NATIVE_CLIENT_HEADER);
        boolean nativeClient = clientMarker != null && !clientMarker.isBlank();

        Map<String, Object> responseBody = new LinkedHashMap<>();
        responseBody.put("username", appUser.getUsername());
        responseBody.put("email", appUser.getEmail());
        responseBody.put("role", appUser.getPlatformRole().name());
        if (nativeClient) {
            responseBody.put("accessToken", jwt);
        }

        return ResponseEntity.ok()
                .header(HttpHeaders.SET_COOKIE, jwtCookie.toString())
                .body(responseBody);
    }

    /**
     * Returns the currently authenticated user's identity. Used by SPAs to
     * bootstrap auth state on page load — the JWT lives in an HttpOnly
     * cookie that JS cannot read, so this endpoint is the only way to
     * answer "am I logged in?".
     */
    @GetMapping("/me")
    public ResponseEntity<Map<String, Object>> currentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication == null || !authentication.isAuthenticated()
                || "anonymousUser".equals(authentication.getPrincipal())) {
            // 401 (not 403) because the SPA polls this endpoint to detect
            // whether the session cookie is still valid. The axios interceptor
            // can ignore this specific 401 to avoid a redirect loop, while
            // still treating 401s on data endpoints as a hard logout signal.
            throw new BadCredentialsException("Not authenticated");
        }

        User appUser = userRepository.findByUsername(authentication.getName())
                .orElseThrow(() -> ResourceNotFoundException.forEntity("User", authentication.getName()));

        return ResponseEntity.ok(Map.of(
                "username", appUser.getUsername(),
                "email", appUser.getEmail(),
                "role", appUser.getPlatformRole().name()
        ));
    }

    /**
     * Clears the auth cookie so the browser stops sending it on subsequent
     * requests. Returns 204; the cookie is set with Max-Age=0 to expire
     * immediately.
     */
    @PostMapping("/logout")
    @Auditable(action = AuditAction.USER_LOGGED_OUT)
    public ResponseEntity<Void> logout(HttpServletRequest http) {
        // SE-8: revoke the current token's jti so it stops working immediately. Clearing the cookie
        // alone leaves the token itself valid until exp — a stolen copy would keep working.
        String jwt = readJwtCookie(http);
        if (jwt != null && !jwt.isEmpty()) {
            try {
                tokenRevocationService.revoke(tokenProvider.getJti(jwt), tokenProvider.getExpiration(jwt));
            } catch (RuntimeException e) {
                log.debug("logout: could not revoke token ({})", e.getClass().getSimpleName());
            }
        }
        ResponseCookie cleared = ResponseCookie.from("jwtToken", "")
                .httpOnly(true)
                .secure(cookieSecure)
                .path("/")
                .maxAge(0)
                .sameSite(cookieSameSite)
                .build();
        return ResponseEntity.noContent()
                .header(HttpHeaders.SET_COOKIE, cleared.toString())
                .build();
    }

    private static String readJwtCookie(HttpServletRequest http) {
        if (http.getCookies() != null) {
            for (jakarta.servlet.http.Cookie c : http.getCookies()) {
                if ("jwtToken".equals(c.getName())) {
                    return c.getValue();
                }
            }
        }
        return null;
    }
}
