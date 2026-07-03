package com.federated.fl_platform_api.config;

import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.AuditingAuthenticationFailureHandler;
import com.federated.fl_platform_api.security.AuditingAuthenticationSuccessHandler;
import com.federated.fl_platform_api.security.InternalApiKeyFilter;
import com.federated.fl_platform_api.security.JwtAuthenticationFilter;
import com.federated.fl_platform_api.security.OrgScopeFilter;
import com.federated.fl_platform_api.service.CustomUserDetailsService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity  // Enables @PreAuthorize / @PostAuthorize on controllers + services
public class SecurityConfig {

    private static final Logger log = LoggerFactory.getLogger(SecurityConfig.class);

    @Autowired
    private CustomUserDetailsService customUserDetailsService;

    @Autowired
    private JwtAuthenticationFilter jwtAuthFilter;

    @Autowired
    private InternalApiKeyFilter internalApiKeyFilter;

    @Autowired
    private OrgScopeFilter orgScopeFilter;

    @Autowired
    private Environment environment;

    @Value("${app.cors.allowed-origins}")
    private String allowedOriginsCsv;

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuditingAuthenticationSuccessHandler auditingAuthenticationSuccessHandler(
            UserRepository users, AuditEventRepository audits) {
        return new AuditingAuthenticationSuccessHandler(users, audits);
    }

    @Bean
    public AuditingAuthenticationFailureHandler auditingAuthenticationFailureHandler(
            AuditEventRepository audits) {
        return new AuditingAuthenticationFailureHandler(audits);
    }

    @Bean
    public DaoAuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(customUserDetailsService);
        authProvider.setPasswordEncoder(passwordEncoder());
        return authProvider;
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        List<String> origins = Arrays.stream(allowedOriginsCsv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .toList();
        if (origins.isEmpty()) {
            throw new IllegalStateException("CORS_ALLOWED_ORIGINS must be set to an explicit, non-empty allowlist");
        }
        // Use allowedOriginPatterns (not allowedOrigins) so entries may contain wildcards
        // (e.g. "http://localhost:*") while still permitting credentials. With credentials
        // enabled, Spring requires patterns instead of a literal "*" origin.
        configuration.setAllowedOriginPatterns(origins);
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(List.of("Authorization", "Content-Type", "Accept", "X-Requested-With"));
        configuration.setAllowCredentials(true);
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        // Public paths (everything else requires authentication). The H2 web
        // console path was removed when H2 was retired in favour of PostgreSQL.
        List<String> publicPaths = new ArrayList<>(List.of(
                "/api/auth/**",
                "/ws-logs/**",
                "/error",
                "/actuator/health"
        ));

        http
                .cors(cors -> cors.configurationSource(corsConfigurationSource()))
                .csrf(csrf -> csrf.disable())
                .headers(headers -> headers.frameOptions(frame -> frame.sameOrigin()))
                .authorizeHttpRequests(authz -> authz
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                        .requestMatchers(publicPaths.toArray(new String[0])).permitAll()
                        // Service-to-service callbacks from FL-server tasks. The chain is permitAll
                        // here because InternalApiKeyFilter (added below) rejects any request without
                        // a valid X-Internal-Key header before Spring Security sees it.
                        .requestMatchers("/api/internal/**").permitAll()
                        // SE-5: actuator management endpoints (loggers/metrics/…) are admin-only —
                        // a plain USER could otherwise POST /actuator/loggers to flip log levels
                        // (log-flood DoS / recon). /actuator/health stays permitAll via publicPaths
                        // above for load-balancer liveness checks.
                        .requestMatchers("/actuator/**").hasRole("PLATFORM_ADMIN")
                        .anyRequest().authenticated()
                )
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authenticationProvider(authenticationProvider())
                .addFilterBefore(internalApiKeyFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class)
                // Runs after auth is established so it can resolve the caller's
                // org memberships and populate the request-scoped OrgScope.
                .addFilterAfter(orgScopeFilter, JwtAuthenticationFilter.class);

        return http.build();
    }
}
