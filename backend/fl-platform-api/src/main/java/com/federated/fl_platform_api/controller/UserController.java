package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.RegisterRequest;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.UserService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;
    private final UserRepository userRepository;

    @Autowired
    public UserController(UserService userService, UserRepository userRepository) {
        this.userService = userService;
        this.userRepository = userRepository;
    }

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<User>> getAllUsers() {
        // Admin-only: previously this leaked the entire user table (id, email,
        // timestamps) to any authenticated user, allowing trivial enumeration.
        // Bootstrap the first admin with:
        //   UPDATE users SET role = 'ADMIN' WHERE username = '<you>';
        return ResponseEntity.ok(userRepository.findAll());
    }

    @PostMapping
    public ResponseEntity<User> createUser(@Valid @RequestBody RegisterRequest request) {
        // UserAlreadyExistsException → 409 via GlobalExceptionHandler.
        User newUser = new User();
        newUser.setUsername(request.getUsername());
        newUser.setEmail(request.getEmail());
        newUser.setPassword(request.getPassword());

        User savedUser = userService.registerUser(newUser);
        return ResponseEntity.status(HttpStatus.CREATED).body(savedUser);
    }

    @DeleteMapping("/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    @SuppressWarnings("null")
    public ResponseEntity<Map<String, String>> deleteUser(@PathVariable Long id) {
        if (!userRepository.existsById(id)) {
            throw ResourceNotFoundException.forEntity("User", id);
        }
        userRepository.deleteById(id);
        return ResponseEntity.ok(Map.of("message", "User deleted successfully"));
    }
}
