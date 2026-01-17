package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Autowired
    private UserRepository userRepository;

    // ...
    @Override
    public UserDetails loadUserByUsername(String usernameOrEmail) throws UsernameNotFoundException {
        System.out.println("LOAD_USER_BY_USERNAME: Input identifier: '" + usernameOrEmail + "'");

        com.federated.fl_platform_api.model.User applicationUser = userRepository.findByUsername(usernameOrEmail)
                .map(u -> {
                    System.out.println("LOAD_USER_BY_USERNAME: Found by username: '" + u.getUsername() + "' (Email: '" + u.getEmail() + "')");
                    return u;
                })
                .orElseGet(() -> {
                    System.out.println("LOAD_USER_BY_USERNAME: Not found by username, trying by email (case-insensitive): '" + usernameOrEmail + "'");
                    return userRepository.findByEmailIgnoreCase(usernameOrEmail)
                            .map(u -> {
                                System.out.println("LOAD_USER_BY_USERNAME: Found by emailIgnoreCase: '" + u.getEmail() + "' (Username: '" + u.getUsername() + "')");
                                return u;
                            })
                            .orElseThrow(() -> {
                                System.err.println("LOAD_USER_BY_USERNAME: User NOT FOUND with identifier: '" + usernameOrEmail + "'");
                                return new UsernameNotFoundException("User not found with username or email: " + usernameOrEmail);
                            });
                });

        // CRITICAL LOG: What is the exact username being passed to Spring Security UserDetails?
        System.out.println("LOAD_USER_BY_USERNAME: Returning UserDetails for DB Username: '" + applicationUser.getUsername() + "' (Length: " + applicationUser.getUsername().length() + ")");

        return new org.springframework.security.core.userdetails.User(
                applicationUser.getUsername(),
                applicationUser.getPassword(),
                new ArrayList<>()
        );
    }
}
