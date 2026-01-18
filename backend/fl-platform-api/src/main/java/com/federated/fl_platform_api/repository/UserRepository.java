package com.federated.fl_platform_api.repository;
import java.util.Optional;

/**

Used to Interact with the users table
 */

import com.federated.fl_platform_api.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepository extends JpaRepository<User, Long> {

    /**
     * Finds a user by their username.
     * @param username the username to search for
     * @return an Optional containing the user if found, or an empty Optional otherwise
     */
    Optional<User> findByUsername(String username);

    /**
     * Finds a user by their email address.
     * @param email the email address to search for
     * @return an Optional containing the user if found, or an empty Optional otherwise
     */
    Optional<User> findByEmail(String email);

    Optional<User> findByEmailIgnoreCase(String email);

    /**
     * Checks if a user exists with the given username.
     * @param username the username to check
     * @return true if a user with the username exists, false otherwise
     */
    Boolean existsByUsername(String username);

    /**
     * Checks if a user exists with the given email address.
     * @param email the email address to check
     * @return true if a user with the email exists, false otherwise
     */
    Boolean existsByEmail(String email);
}
