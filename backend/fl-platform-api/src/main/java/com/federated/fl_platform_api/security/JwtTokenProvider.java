package com.federated.fl_platform_api.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.io.Decoders;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;

import java.util.Date;
import java.util.function.Function;

import io.jsonwebtoken.security.Keys;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Component;
import io.jsonwebtoken.Jwts;
import org.springframework.security.core.userdetails.UserDetails;


@Component
public class JwtTokenProvider {

    @Value("${app.jwt.secret}")
    private String jwtSecretString;

    @Value("${app.jwt.expiration-ms}")
    private long jwtExpirationInMs;

    private javax.crypto.SecretKey jwtSecretKey;

    @PostConstruct
    public void init() {
        byte[] keyBytes = Decoders.BASE64.decode(jwtSecretString);
        this.jwtSecretKey = Keys.hmacShaKeyFor(keyBytes);
    }

    public String generateToken(Authentication authentication) {
        UserDetails userPrincipal = (UserDetails) authentication.getPrincipal();

        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + jwtExpirationInMs);

        return Jwts.builder()
                .subject(userPrincipal.getUsername())
                .issuedAt(now)
                .expiration(expiryDate)
                .signWith(jwtSecretKey)
                .compact();
    }

    public String getUsernameFromToken(String token) {
        return extractClaim(token, Claims::getSubject);
    }


    public boolean validateToken(String token, UserDetails userDetails) {
        final String username = getUsernameFromToken(token);
        return (username.equals(userDetails.getUsername())) && !isTokenExpired(token);
    }

    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }



    private Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }

    private <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parser()
                .verifyWith(jwtSecretKey)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

}
