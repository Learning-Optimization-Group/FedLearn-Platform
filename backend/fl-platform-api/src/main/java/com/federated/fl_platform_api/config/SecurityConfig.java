package com.federated.fl_platform_api.config;

import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.federated.fl_platform_api.security.AuditingAuthenticationFailureHandler;
import com.federated.fl_platform_api.security.AuditingAuthenticationSuccessHandler;
import com.federated.fl_platform_api.security.InternalApiKeyFilter;
import com.federated.fl_platform_api.security.JwtAuthenticationFilter;
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
        // Build the public-path list dynamically. /h2-console/** is only added
        // when the dev profile is active so that an accidentally-running prod
        // instance can never expose the H2 web console — even if a future
        // migration toggles spring.h2.console.enabled.
        List<String> publicPaths = new ArrayList<>(List.of(
                "/api/auth/**",
                "/ws-logs/**",
                "/error",
                "/actuator/health"
        ));
        if (environment.acceptsProfiles(profiles -> profiles.test("dev"))) {
            publicPaths.add("/h2-console/**");
            log.warn("DEV PROFILE ACTIVE — /h2-console/** is publicly accessible. Do not run this profile in production.");
        }

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
                        .anyRequest().authenticated()
                )
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authenticationProvider(authenticationProvider())
                .addFilterBefore(internalApiKeyFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
