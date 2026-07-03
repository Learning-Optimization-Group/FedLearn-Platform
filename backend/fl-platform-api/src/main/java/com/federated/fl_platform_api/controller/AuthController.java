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
import com.federated.fl_platform_api.security.JwtTokenProvider;
import com.federated.fl_platform_api.security.LoginRateLimiter;
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

import java.util.Map;

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

    @Value("${app.auth.cookie.secure:true}")
    private boolean cookieSecure;

    @Value("${app.auth.cookie.same-site:Strict}")
    private String cookieSameSite;

    @Value("${app.auth.cookie.max-age-seconds:3600}")
    private long cookieMaxAgeSeconds;

    @Autowired
    public AuthController(UserService userService, AuthenticationManager authenticationManager,
                          JwtTokenProvider tokenProvider, UserRepository userRepository,
                          AuditingAuthenticationSuccessHandler successHandler,
                          AuditingAuthenticationFailureHandler failureHandler,
                          LoginRateLimiter loginRateLimiter) {
        this.userService = userService;
        this.authenticationManager = authenticationManager;
        this.tokenProvider = tokenProvider;
        this.userRepository = userRepository;
        this.successHandler = successHandler;
        this.failureHandler = failureHandler;
        this.loginRateLimiter = loginRateLimiter;
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
            log.warn("Login throttled for user '{}' from {}", loginRequest.getUsername(), http.getRemoteAddr());
            return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
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
                .maxAge(cookieMaxAgeSeconds)
                .sameSite(cookieSameSite)
                .build();

        // The JWT is set as an HttpOnly cookie (defeats XSS exfiltration) for browser SPAs,
        // which ignore the body and rely on the cookie + GET /api/auth/me for session checks.
        // The body also carries accessToken for native clients (mobile/desktop) that cannot
        // reliably read the HttpOnly Set-Cookie header — they store it in secure platform
        // storage (e.g. Keychain / EncryptedSharedPreferences) and send it as a Bearer token.
        Map<String, Object> responseBody = Map.of(
                "username", appUser.getUsername(),
                "email", appUser.getEmail(),
                "role", appUser.getPlatformRole().name(),
                "accessToken", jwt
        );

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
    public ResponseEntity<Void> logout() {
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
}
