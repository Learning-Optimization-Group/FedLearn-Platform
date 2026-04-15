package com.federated.fl_platform_api.dto;

import lombok.Data;
import java.time.Instant;

@Data
public class ServerLogDto {
    private String level;
    private String message;
    private String stackTrace;
    private Instant timestamp;
}
