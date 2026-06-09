package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.DisabledException;
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

        // Lifecycle gate: refuse to authenticate users that aren't active. We throw
        // before returning the UserDetails so the ProviderManager treats this the
        // same as any other AuthenticationException — the controller's failure
        // handler still fires and the response is a 401. Performed before the
        // password check, which is acceptable because we don't leak account
        // existence (the failure path is identical to bad-credentials).
        if (applicationUser.getStatus() != UserStatus.ACTIVE || applicationUser.getDeletedAt() != null) {
            log.info("Authentication failed: user is not in ACTIVE status");
            throw new DisabledException("inactive");
        }

        // Spring Security expects authorities prefixed with "ROLE_" for the
        // hasRole(...) DSL to match. We store the bare role on the entity
        // ("USER" / "ADMIN") and prefix it here at the boundary.
        String role = applicationUser.getPlatformRole() != null ? applicationUser.getPlatformRole() : "USER";
        SimpleGrantedAuthority authority = new SimpleGrantedAuthority("ROLE_" + role);

        return new org.springframework.security.core.userdetails.User(
                applicationUser.getUsername(),
                applicationUser.getPassword(),
                List.of(authority)
        );
    }
}
