package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public class ChatTurn {
    @Pattern(regexp = "user|assistant") private String role;
    @NotBlank @Size(max = 10_000) private String content;
    public String getRole() { return role; }
    public void setRole(String role) { this.role = role; }
    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }
}
