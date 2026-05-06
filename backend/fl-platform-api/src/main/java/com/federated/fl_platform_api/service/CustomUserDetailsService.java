package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    private static final Logger log = LoggerFactory.getLogger(CustomUserDetailsService.class);

    @Autowired
    private UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String usernameOrEmail) throws UsernameNotFoundException {
        // Avoid logging the raw identifier at INFO — emails are PII.
        log.debug("loadUserByUsername invoked");

        com.federated.fl_platform_api.model.User applicationUser = userRepository.findByUsername(usernameOrEmail)
                .orElseGet(() -> userRepository.findByEmailIgnoreCase(usernameOrEmail)
                        .orElseThrow(() -> {
                            // Translates to a 401 via the auth handler in GlobalExceptionHandler.
                            log.info("Authentication failed: identifier not found");
                            return new UsernameNotFoundException(
                                    "User not found with username or email: " + usernameOrEmail);
                        }));

        // Spring Security expects authorities prefixed with "ROLE_" for the
        // hasRole(...) DSL to match. We store the bare role on the entity
        // ("USER" / "ADMIN") and prefix it here at the boundary.
        String role = applicationUser.getRole() != null ? applicationUser.getRole() : "USER";
        SimpleGrantedAuthority authority = new SimpleGrantedAuthority("ROLE_" + role);

        return new org.springframework.security.core.userdetails.User(
                applicationUser.getUsername(),
                applicationUser.getPassword(),
                List.of(authority)
        );
    }
}
