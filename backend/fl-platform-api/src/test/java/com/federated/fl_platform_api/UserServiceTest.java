package com.federated.fl_platform_api;

import com.federated.fl_platform_api.exception.UserAlreadyExistsException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.service.UserService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @InjectMocks
    private UserService userService;

    private User buildRequest(String username, String email) {
        User u = new User();
        u.setUsername(username);
        u.setEmail(email);
        u.setPassword("plaintext");
        return u;
    }

    @Test
    void registerUser_withNewCredentials_shouldSaveAndReturnUser() {
        when(userRepository.existsByUsername("alice")).thenReturn(false);
        when(userRepository.existsByEmail("alice@example.com")).thenReturn(false);
        when(passwordEncoder.encode("plaintext")).thenReturn("$hashed$");
        when(userRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

        User result = userService.registerUser(buildRequest("alice", "alice@example.com"));

        assertNotNull(result);
        assertEquals("alice", result.getUsername());
        assertEquals("$hashed$", result.getPassword());
        verify(userRepository).save(any());
    }

    @Test
    void registerUser_withDuplicateUsername_shouldThrowUserAlreadyExistsException() {
        when(userRepository.existsByUsername("alice")).thenReturn(true);

        assertThrows(UserAlreadyExistsException.class,
                () -> userService.registerUser(buildRequest("alice", "other@example.com")));

        verify(userRepository, never()).save(any());
    }

    @Test
    void registerUser_withDuplicateEmail_shouldThrowUserAlreadyExistsException() {
        when(userRepository.existsByUsername("newuser")).thenReturn(false);
        when(userRepository.existsByEmail("alice@example.com")).thenReturn(true);

        assertThrows(UserAlreadyExistsException.class,
                () -> userService.registerUser(buildRequest("newuser", "alice@example.com")));

        verify(userRepository, never()).save(any());
    }
}
