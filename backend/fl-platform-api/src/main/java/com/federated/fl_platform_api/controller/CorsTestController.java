package com.federated.fl_platform_api.controller;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class CorsTestController {

    @RequestMapping(value = "/api/auth/register", method = RequestMethod.OPTIONS)
    public ResponseEntity<?> corsPreflightTest(HttpServletRequest request) {
        System.out.println("✅ OPTIONS /api/auth/register hit");
        HttpHeaders headers = new HttpHeaders();
        headers.add("Access-Control-Allow-Origin", request.getHeader("Origin"));
        headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        headers.add("Access-Control-Allow-Headers", request.getHeader("Access-Control-Request-Headers"));
        headers.add("Access-Control-Allow-Credentials", "true");
        return new ResponseEntity<>(headers, HttpStatus.NO_CONTENT);
    }
}
