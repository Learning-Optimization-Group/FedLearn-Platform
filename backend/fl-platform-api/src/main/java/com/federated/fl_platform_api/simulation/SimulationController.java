package com.federated.fl_platform_api.simulation;

import org.springframework.http.*;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;

@RestController
@RequestMapping("/api/simulations")
@PreAuthorize("hasAnyRole('PROJECT_OWNER','PLATFORM_ADMIN')")
public class SimulationController {
    private final SimulationService service;

    public SimulationController(SimulationService service) { this.service = service; }

    @GetMapping("/workloads") public SimulationDtos.Workloads workloads() { return service.workloads(); }
    @PostMapping("/resolve") public SimulationDtos.Resolved resolve(HttpServletRequest request) throws IOException {
        return service.resolve(SimulationJson.readChoices(request.getInputStream()));
    }
    @PostMapping("/runs") public ResponseEntity<SimulationDtos.Accepted> start(HttpServletRequest request) throws IOException {
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(service.start(SimulationJson.readChoices(request.getInputStream())));
    }
    @GetMapping("/runs") public SimulationDtos.Page list(@RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size, @RequestParam(defaultValue = "mine") String scope) {
        return service.list(page, size, scope);
    }
    @GetMapping("/runs/{id}") public SimulationDtos.Detail get(@PathVariable UUID id) { return service.get(id); }
    @PostMapping("/runs/{id}/stop") public SimulationDtos.Detail stop(@PathVariable UUID id) { return service.stop(id); }
    @GetMapping("/runs/{id}/record") public ResponseEntity<?> download(@PathVariable UUID id) {
        return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON)
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"simulation-" + id + "-research.json\"")
                .header(HttpHeaders.CACHE_CONTROL, "no-store").body(service.download(id));
    }
    @ExceptionHandler(SimulationException.class) public ResponseEntity<?> error(SimulationException error) {
        return ResponseEntity.status(error.status).body(Map.of("code", error.code, "message", error.getMessage()));
    }
}
