package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
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

        if(userRepository.existsByUsername(userRequest.getUsername())) {
            throw new UserAlreadyExistsException("Username " + userRequest.getUsername() + " is already taken.");
        }
        if(userRepository.existsByEmail(userRequest.getEmail())) {
            throw new UserAlreadyExistsException("Email " + userRequest.getEmail() + " is already registered.");
        }

        User newUser = new User();
        newUser.setUsername(userRequest.getUsername());
        newUser.setEmail(userRequest.getEmail());

        newUser.setPassword(passwordEncoder.encode(userRequest.getPassword()));

        // Set timestamps (if not handled by JPA Auditing yet)
        Instant now = Instant.now();
        newUser.setCreatedAt(now);
        newUser.setUpdatedAt(now);

        // Save the new user
        return userRepository.save(newUser);
    }

    public static void main(String args[]){
        // In a test or a temporary main method
        PasswordEncoder encoder = new BCryptPasswordEncoder();
        String plainPassword = "123456";
        String hashedPasswordFromDB = "$2a$10$q6Hkmk3sOTNRpAkp73.oJu6ONJQA0hfMUwBDxNMBMPEyjUe17MPDa"; // Paste actual hash

        boolean matches = encoder.matches(plainPassword, hashedPasswordFromDB);
        System.out.println("Password matches: " + matches);
    }


}
