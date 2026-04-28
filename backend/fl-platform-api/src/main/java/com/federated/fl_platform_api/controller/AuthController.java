package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.LoginRequest;
import com.federated.fl_platform_api.dto.RegisterRequest;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.JwtTokenProvider;
import com.federated.fl_platform_api.service.UserService;
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

    @Value("${app.auth.cookie.secure:true}")
    private boolean cookieSecure;

    @Value("${app.auth.cookie.same-site:Strict}")
    private String cookieSameSite;

    @Value("${app.auth.cookie.max-age-seconds:3600}")
    private long cookieMaxAgeSeconds;

    @Autowired
    public AuthController(UserService userService, AuthenticationManager authenticationManager,
                          JwtTokenProvider tokenProvider, UserRepository userRepository) {
        this.userService = userService;
        this.authenticationManager = authenticationManager;
        this.tokenProvider = tokenProvider;
        this.userRepository = userRepository;
    }

    @PostMapping("/register")
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
    public ResponseEntity<Map<String, Object>> authenticateUser(@Valid @RequestBody LoginRequest loginRequest) {
        // AuthenticationException (bad credentials, locked, etc.) → 401 via GlobalExceptionHandler.
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        loginRequest.getUsername(),
                        loginRequest.getPassword()
                )
        );

        String authenticatedPrincipalName = authentication.getName();
        SecurityContextHolder.getContext().setAuthentication(authentication);
        String jwt = tokenProvider.generateToken(authentication);

        User appUser = userRepository.findByUsername(authenticatedPrincipalName)
                .orElseThrow(() -> {
                    // Should never happen — auth succeeded but the user row vanished mid-request.
                    log.error("Authenticated principal '{}' has no matching user row", authenticatedPrincipalName);
                    return ResourceNotFoundException.forEntity("User", authenticatedPrincipalName);
                });

        ResponseCookie jwtCookie = ResponseCookie.from("jwtToken", jwt)
                .httpOnly(true)
                .secure(cookieSecure)
                .path("/")
                .maxAge(cookieMaxAgeSeconds)
                .sameSite(cookieSameSite)
                .build();

        // Cookie-only auth: the JWT lives exclusively in the HttpOnly cookie so
        // it can never be read by JS (defeats XSS exfiltration). The body
        // returns only the user identity the SPA needs to render the shell.
        // Frontends should call GET /api/auth/me on bootstrap to learn whether
        // a session cookie is still valid.
        Map<String, Object> responseBody = Map.of(
                "username", appUser.getUsername(),
                "email", appUser.getEmail(),
                "role", appUser.getRole()
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
                "role", appUser.getRole()
        ));
    }

    /**
     * Clears the auth cookie so the browser stops sending it on subsequent
     * requests. Returns 204; the cookie is set with Max-Age=0 to expire
     * immediately.
     */
    @PostMapping("/logout")
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
