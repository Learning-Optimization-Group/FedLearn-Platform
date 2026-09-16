package com.federated.fl_platform_api.repository;
import java.util.Optional;

/**

Used to Interact with the users table
 */

import com.federated.fl_platform_api.model.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

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

    long countByPlatformRole(com.federated.fl_platform_api.model.PlatformRole platformRole);

    List<User> findByPlatformRole(com.federated.fl_platform_api.model.PlatformRole platformRole);

    /**
     * Existence check used by the bootstrap runner to short-circuit when the
     * first platform admin has already been seeded.
     */
    boolean existsByPlatformRole(com.federated.fl_platform_api.model.PlatformRole platformRole);

    List<User> findByUsernameStartingWithIgnoreCaseOrderByUsernameAsc(String prefix, Pageable pageable);

    /**
     * Admin directory search: case-insensitive substring match on username OR
     * email, optionally narrowed by platform role and account status. Sorting
     * comes from the {@link Pageable} (the admin endpoint sorts username asc).
     *
     * <p>{@code pattern} must be a non-null, pre-lowercased LIKE pattern
     * (caller passes {@code "%"} for match-all) — binding a nullable string
     * through {@code LOWER(CONCAT(...))} makes Postgres infer {@code bytea}
     * and fail with "function lower(bytea) does not exist".
     */
    @Query("""
        SELECT u FROM User u
        WHERE (LOWER(u.username) LIKE :pattern
               OR LOWER(u.email) LIKE :pattern)
          AND (:role   IS NULL OR u.platformRole = :role)
          AND (:status IS NULL OR u.status       = :status)
        """)
    Page<User> searchForAdmin(
            @Param("pattern") String pattern,
            @Param("role") com.federated.fl_platform_api.model.PlatformRole role,
            @Param("status") com.federated.fl_platform_api.model.UserStatus status,
            Pageable pageable);

    /**
     * Count used by the suspension guard: the platform must always keep at
     * least one ACTIVE PLATFORM_ADMIN.
     */
    long countByPlatformRoleAndStatus(com.federated.fl_platform_api.model.PlatformRole platformRole,
                                      com.federated.fl_platform_api.model.UserStatus status);
}
