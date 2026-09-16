package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.DeviceRequirements;
import com.federated.fl_platform_api.model.Project;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class RequirementsService {

    @Autowired private ModelRecipeService modelRecipeService;

    /** Effective requirements = recipe default merged most-restrictive-wins with the
     *  project's optional override. Null recipe + null override => null. */
    public DeviceRequirements effectiveFor(Project project) {
        DeviceRequirements recipeDefault = modelRecipeService.findByKey(project.getModelType())
                .map(r -> r.requirements())
                .orElse(null);
        DeviceRequirements override = project.getRequirementsOverride();
        if (recipeDefault == null) return override;
        return DeviceRequirements.merge(recipeDefault, override);
    }
}
