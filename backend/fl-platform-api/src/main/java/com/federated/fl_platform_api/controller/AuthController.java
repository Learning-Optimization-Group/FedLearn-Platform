package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.LoginRequest;
import com.federated.fl_platform_api.dto.RegisterRequest;
import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.JwtTokenProvider;
import com.federated.fl_platform_api.service.UserService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.security.core.AuthenticationException;

import java.util.HashMap;
import java.util.Map;
import com.federated.fl_platform_api.security.JwtTokenProvider;
import org.springframework.security.core.userdetails.UserDetails;

@RestController
@RequestMapping("/api/auth")
// @CrossOrigin(origins = "*") // Keep if you don't have global WebConfig, remove if you do
public class AuthController {

    private final UserService userService;
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider tokenProvider;
    private final UserRepository userRepository;


    @Autowired
    public AuthController(UserService userService, AuthenticationManager authenticationManager, JwtTokenProvider tokenProvider, UserRepository userRepository) {
        this.userService = userService;
        this.authenticationManager = authenticationManager;
        this.tokenProvider = tokenProvider;
        this.userRepository = userRepository;
    }

    @PostMapping("/register")
    public ResponseEntity<?> registerUser(@Valid @RequestBody RegisterRequest registerRequest) {
        try {
            User newUser = new User();
            newUser.setUsername(registerRequest.getUsername());
            newUser.setEmail(registerRequest.getEmail());
            newUser.setPassword(registerRequest.getPassword());

            User registeredUser = userService.registerUser(newUser);

            // Return a structured JSON response
            Map<String, Object> responseBody = Map.of(
                    "message", "User registered successfully!",
                    "userId", registeredUser.getId() // Or any other relevant info
            );
            return ResponseEntity.status(HttpStatus.CREATED).contentType(MediaType.APPLICATION_JSON).body(responseBody);

        } catch (UserAlreadyExistsException e) {
            // Return error as JSON too for consistency
            Map<String, String> errorBody = Map.of("error", e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).contentType(MediaType.APPLICATION_JSON).body(errorBody);
        } catch (Exception e) {
            e.printStackTrace(); // Good to log the full stack trace for unexpected errors
            Map<String, String> errorBody = Map.of("error", "An unexpected error occurred. Please try again.");
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).contentType(MediaType.APPLICATION_JSON).body(errorBody);
        }
    }

    // AuthController.java
    @PostMapping("/login")
    public ResponseEntity<?> authenticateUser(@Valid @RequestBody LoginRequest loginRequest) {
        try {
            System.out.println("AUTH_CTRL_LOGIN: Identifier from request: '" + loginRequest.getUsername() + "'");
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            loginRequest.getUsername(),
                            loginRequest.getPassword()
                    )
            );

            String authenticatedPrincipalName = authentication.getName(); // This is UserDetails.getUsername()
            System.out.println("AUTH_CTRL_LOGIN: Authentication successful. Principal name from Spring Security: '" + authenticatedPrincipalName + "' (Length: " + authenticatedPrincipalName.length() + ")");

            SecurityContextHolder.getContext().setAuthentication(authentication);
            String jwt = tokenProvider.generateToken(authentication);
            System.out.println("AUTH_CTRL_LOGIN: JWT generated.");

            System.out.println("AUTH_CTRL_LOGIN_FETCH: Attempting to fetch User entity from repository using username: '" + authenticatedPrincipalName + "'");

            User appUser = userRepository.findByUsername(authenticatedPrincipalName) // This is a case-sensitive lookup
                    .orElseThrow(() -> {
                        String errorMsg = "CRITICAL_ERROR: User entity for username '" + authenticatedPrincipalName + "' not found in repository AFTER successful authentication.";
                        System.err.println(errorMsg);
                        // Log details of the principal if it helps
                        Object principalObj = authentication.getPrincipal();
                        if (principalObj instanceof UserDetails) {
                            UserDetails ud = (UserDetails) principalObj;
                            System.err.println("UserDetails principal details at time of error: Username='" + ud.getUsername() + "', Enabled=" + ud.isEnabled());
                        } else {
                            System.err.println("Principal is not UserDetails at time of error: " + principalObj.toString());
                        }
                        return new RuntimeException(errorMsg + " Identifier was: " + loginRequest.getUsername());
                    });

            System.out.println("AUTH_CTRL_LOGIN_FETCH_SUCCESS: User entity fetched: Username='" + appUser.getUsername() + "', Email='" + appUser.getEmail() + "'");

            Map<String, Object> responseBody = new HashMap<>();
            responseBody.put("accessToken", jwt);
            responseBody.put("tokenType", "Bearer");
            responseBody.put("username", appUser.getUsername());
            responseBody.put("email", appUser.getEmail());

            return ResponseEntity.ok(responseBody);

        } catch (AuthenticationException e) {
            System.err.println("AUTH_CTRL_LOGIN_FAILURE: AuthenticationException: " + e.getMessage() + " for identifier: " + loginRequest.getUsername());
            Map<String, String> errorBody = Map.of("error", "Login failed: Invalid username or password.");
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorBody);
        }
    }




}