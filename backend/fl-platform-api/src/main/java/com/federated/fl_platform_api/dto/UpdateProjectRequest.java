package com.federated.fl_platform_api.dto;

import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public class UpdateProjectRequest {
    @Size(max = 255)
    private String name;

    @Size(max = 4000)
    private String description;

    @Pattern(regexp = "PUBLIC|PRIVATE")
    private String visibility;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }
}
