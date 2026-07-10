package com.federated.fl_platform_api.dto;

import com.federated.fl_platform_api.model.ProjectVisibility;
import com.federated.fl_platform_api.validation.ValueOfEnum;
import jakarta.validation.constraints.Size;

public class UpdateProjectRequest {
    @Size(max = 255)
    private String name;

    @Size(max = 4000)
    private String description;

    // Validated against the ProjectVisibility enum so the accepted set can never
    // drift from the source of truth (BA-15). null = "leave visibility unchanged".
    @ValueOfEnum(enumClass = ProjectVisibility.class)
    private String visibility;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getVisibility() { return visibility; }
    public void setVisibility(String visibility) { this.visibility = visibility; }

    private DeviceRequirements requirementsOverride;
    public DeviceRequirements getRequirementsOverride() { return requirementsOverride; }
    public void setRequirementsOverride(DeviceRequirements requirementsOverride) { this.requirementsOverride = requirementsOverride; }
}
