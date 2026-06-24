package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.DeviceRequirements;
import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.service.ModelRecipeService;
import com.federated.fl_platform_api.service.RequirementsService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.Optional;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RequirementsServiceTest {

    @Mock ModelRecipeService recipeService;
    @InjectMocks RequirementsService service;

    private ModelRecipeDto recipeWith(DeviceRequirements req) {
        return new ModelRecipeDto("TRANSFORMER", "T", "text", List.of(), List.of(), List.of(), req);
    }
    private Project project(DeviceRequirements override) {
        Project p = new Project();
        p.setModelType("TRANSFORMER");
        p.setRequirementsOverride(override);
        return p;
    }

    @Test
    void effectiveFor_mergesRecipeAndOverride() {
        DeviceRequirements recipe = new DeviceRequirements(8.0, 1.5, null, null, Boolean.FALSE,
                125_000_000L, null, null, null, null, null);
        when(recipeService.findByKey("TRANSFORMER")).thenReturn(Optional.of(recipeWith(recipe)));
        DeviceRequirements override = new DeviceRequirements(12.0, null, null, null, null,
                null, null, null, null, null, null);

        DeviceRequirements eff = service.effectiveFor(project(override));
        assertEquals(12.0, eff.minRamGb());                 // tightened
        assertEquals(125_000_000L, eff.maxTrainableParams()); // recipe-only preserved
        assertEquals(Boolean.FALSE, eff.mobileSafe());
    }

    @Test
    void effectiveFor_noOverrideReturnsRecipe() {
        DeviceRequirements recipe = new DeviceRequirements(4.0, 0.2, null, null, Boolean.TRUE,
                null, null, null, null, null, null);
        when(recipeService.findByKey("TRANSFORMER")).thenReturn(Optional.of(recipeWith(recipe)));
        DeviceRequirements eff = service.effectiveFor(project(null));
        assertEquals(4.0, eff.minRamGb());
    }

    @Test
    void effectiveFor_unknownRecipeReturnsOverrideOrNull() {
        when(recipeService.findByKey("TRANSFORMER")).thenReturn(Optional.empty());
        assertNull(service.effectiveFor(project(null)));
    }
}
