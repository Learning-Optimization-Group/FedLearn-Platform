# 02 - Security and Authentication

The FedLearn backend implements a robust, multi-layered security architecture designed to handle both standard REST API clients (React) and internal Machine Learning servers (Python).

## 1. REST API Security (JWT)

The platform uses **JSON Web Tokens (JWT)** for stateless user authentication. When a user logs in via `/api/auth/login`, the `AuthController` generates a token using `JwtTokenProvider` and returns it to the client.

### Token Extraction Strategy

The `JwtAuthenticationFilter` intercepts every incoming HTTP request and attempts to extract the token from two potential sources in priority order:

```java
String authHeader = request.getHeader("Authorization");
String jwt = null;

// 1. Check Authorization Header (used by programmatic clients)
if (authHeader != null && authHeader.startsWith("Bearer ")) {
    jwt = authHeader.substring(7);
} 
// 2. Check HttpOnly Cookies (used securely by the browser)
else if (request.getCookies() != null) {
    for (jakarta.servlet.http.Cookie cookie : request.getCookies()) {
        if ("jwtToken".equals(cookie.getName())) {
            jwt = cookie.getValue();
            break;
        }
    }
}
```

By supporting both mechanisms, the backend is highly flexible while remaining secure against Cross-Site Scripting (XSS) when using `HttpOnly` cookies on the web.

---

## 2. WebSocket Security (STOMP)

WebSockets present a unique security challenge because the initial connection is an HTTP Upgrade request, but subsequent messages flow over a persistent, non-HTTP TCP channel.

We secure WebSockets at **two distinct layers**:

### Layer 1: Handshake Interception (`JwtHandshakeInterceptor.java`)

This is the most critical defense. Before the WebSocket channel is even opened, the `JwtHandshakeInterceptor` validates the token on the initial HTTP upgrade request.

If the token is invalid or missing, the handshake is rejected with a `401 Unauthorized` status, completely preventing an unauthenticated client from opening a socket.

```java
// Inside JwtHandshakeInterceptor.java
@Override
public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                               WebSocketHandler wsHandler, Map<String, Object> attributes) {
    String token = extractToken(request);
    if (token == null || !jwtTokenProvider.validateToken(token, userDetails)) {
        reject(response);
        return false; // Connection dropped before socket opens
    }
    
    // Store authenticated principal in attributes to pass to STOMP layer
    attributes.put(PRINCIPAL_ATTR, auth);
    return true; 
}
```

### Layer 2: STOMP Channel Interception (`JwtChannelInterceptor.java`)

Once the socket is open, STOMP clients send `CONNECT` frames. The `JwtChannelInterceptor` takes the `PRINCIPAL_ATTR` saved during the handshake and officially promotes it into the STOMP message header context. This ensures that any `@SubscribeMapping` or message routing logic has access to the user's identity.

---

## 3. Machine-to-Machine Security (Internal API Key)

The Python Federated Learning server (spawned by AWS ECS or `ProcessBuilder`) needs to send training results and status updates back to the Spring Boot API.

**Problem:** The Python script is not a "User". It doesn't have a username/password, and generating temporary JWTs for the script introduces unnecessary complexity.

**Solution:** The `InternalApiKeyFilter`.

All internal reporting endpoints are placed under `/api/internal/**`. The `SecurityConfig` allows these endpoints to bypass the standard JWT checks. Instead, they are intercepted by the `InternalApiKeyFilter`.

```java
// Inside InternalApiKeyFilter.java
@Override
protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                FilterChain filterChain) throws ServletException, IOException {
    
    String requestApiKey = request.getHeader("X-Internal-API-Key");

    if (internalApiKey.equals(requestApiKey)) {
        // Authenticate the request as an internal service
        UsernamePasswordAuthenticationToken authentication =
                new UsernamePasswordAuthenticationToken("internal-service", null,
                        Collections.singletonList(new SimpleGrantedAuthority("ROLE_INTERNAL")));
        SecurityContextHolder.getContext().setAuthentication(authentication);
    } else {
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        return;
    }
    
    filterChain.doFilter(request, response);
}
```

The Spring Boot backend securely passes this `FEDLEARN_INTERNAL_API_KEY` to the Python server via environment variables during the orchestration phase (see `FlowerServerManager`), ensuring that only backend-spawned ML processes can hit the internal endpoints.
