package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.LoginRequest;
import com.federated.fl_platform_api.dto.RegisterRequest;
import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
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
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
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
    @SuppressWarnings("null")
    public ResponseEntity<?> registerUser(@Valid @RequestBody RegisterRequest registerRequest) {
        try {
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

        } catch (UserAlreadyExistsException e) {
            Map<String, String> errorBody = Map.of("error", e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(errorBody);
        } catch (Exception e) {
            log.error("Unexpected error during user registration", e);
            Map<String, String> errorBody = Map.of("error", "An unexpected error occurred. Please try again.");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(errorBody);
        }
    }

    @PostMapping("/login")
    @SuppressWarnings("null")
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody LoginRequest loginRequest) {
        try {
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
                        log.error("User entity not found in repository after successful authentication for principal");
                        return new RuntimeException("User account is inconsistent. Please contact support.");
                    });

            ResponseCookie jwtCookie = ResponseCookie.from("jwtToken", jwt)
                    .httpOnly(true)
                    .secure(cookieSecure)
                    .path("/")
                    .maxAge(cookieMaxAgeSeconds)
                    .sameSite(cookieSameSite)
                    .build();

            Map<String, Object> responseBody = new HashMap<>();
            responseBody.put("username", appUser.getUsername());
            responseBody.put("email", appUser.getEmail());
            responseBody.put("accessToken", jwt);

            return ResponseEntity.ok()
                    .header(HttpHeaders.SET_COOKIE, jwtCookie.toString())
                    .body(responseBody);

        } catch (AuthenticationException e) {
            log.info("Authentication failed: {}", e.getClass().getSimpleName());
            Map<String, String> errorBody = Map.of("error", "Login failed: Invalid username or password.");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorBody);
        }
    }
}
