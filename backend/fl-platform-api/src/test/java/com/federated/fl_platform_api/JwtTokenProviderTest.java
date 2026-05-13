package com.federated.fl_platform_api;

import com.federated.fl_platform_api.security.JwtTokenProvider;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.test.context.ActiveProfiles;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
class JwtTokenProviderTest {

    @Autowired
    private JwtTokenProvider jwtTokenProvider;

    private Authentication authentication;
    private UserDetails userDetails;

    @BeforeEach
    void setUp() {
        userDetails = User.withUsername("alice")
                .password("password")
                .authorities(Collections.emptyList())
                .build();
        authentication = new UsernamePasswordAuthenticationToken(
                userDetails, null, Collections.emptyList());
    }

    @Test
    void generateToken_shouldReturnNonBlankString() {
        String token = jwtTokenProvider.generateToken(authentication);
        assertNotNull(token);
        assertFalse(token.isBlank());
    }

    @Test
    void getUsernameFromToken_shouldReturnCorrectUsername() {
        String token = jwtTokenProvider.generateToken(authentication);
        String username = jwtTokenProvider.getUsernameFromToken(token);
        assertEquals("alice", username);
    }

    @Test
    void validateToken_withCorrectUser_shouldReturnTrue() {
        String token = jwtTokenProvider.generateToken(authentication);
        boolean valid = jwtTokenProvider.validateToken(token, userDetails);
        assertTrue(valid);
    }

    @Test
    void validateToken_withWrongUser_shouldReturnFalse() {
        String token = jwtTokenProvider.generateToken(authentication);
        UserDetails otherUser = User.withUsername("bob")
                .password("pass")
                .authorities(Collections.emptyList())
                .build();
        boolean valid = jwtTokenProvider.validateToken(token, otherUser);
        assertFalse(valid);
    }
}
