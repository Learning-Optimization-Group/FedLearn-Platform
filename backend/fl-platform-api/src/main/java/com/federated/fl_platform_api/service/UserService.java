package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    @Autowired
    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public User registerUser(User userRequest) {
        if (userRepository.existsByUsername(userRequest.getUsername())) {
            throw new UserAlreadyExistsException(
                    "Username " + userRequest.getUsername() + " is already taken.");
        }
        if (userRepository.existsByEmail(userRequest.getEmail())) {
            throw new UserAlreadyExistsException(
                    "Email " + userRequest.getEmail() + " is already registered.");
        }

        User newUser = new User();
        newUser.setUsername(userRequest.getUsername());
        newUser.setEmail(userRequest.getEmail());
        newUser.setPassword(passwordEncoder.encode(userRequest.getPassword()));

        Instant now = Instant.now();
        newUser.setCreatedAt(now);
        newUser.setUpdatedAt(now);

        return userRepository.save(newUser);
    }
}
