package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.UserSearchResultDto;
import com.federated.fl_platform_api.service.UserSearchService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users/search")
public class UserSearchController {

    @Autowired private UserSearchService searchService;

    @GetMapping
    public ResponseEntity<List<UserSearchResultDto>> search(@RequestParam("q") String q) {
        return ResponseEntity.ok(searchService.search(q));
    }
}
