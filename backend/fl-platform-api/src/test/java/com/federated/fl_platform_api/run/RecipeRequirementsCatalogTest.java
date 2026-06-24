package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.service.ModelRecipeService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
class RecipeRequirementsCatalogTest {

    @Autowired
    ModelRecipeService service;

    @Test
    void everyRecipeDeclaresRequirements() {
        List<ModelRecipeDto> recipes = service.getRecipes();
        assertFalse(recipes.isEmpty());
        for (ModelRecipeDto r : recipes) {
            assertNotNull(r.requirements(), "recipe " + r.key() + " missing requirements");
            assertNotNull(r.requirements().minRamGb(), "recipe " + r.key() + " missing minRamGb");
        }
    }

    @Test
    void transformerIsNotMobileSafe() {
        ModelRecipeDto t = service.getRecipes().stream()
                .filter(r -> "TRANSFORMER".equals(r.key()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("TRANSFORMER recipe not found in catalog"));
        assertEquals(Boolean.FALSE, t.requirements().mobileSafe());
    }

    @Test
    void llmLoraGatesPhonesAndRequiresEightGb() {
        ModelRecipeDto r = service.getRecipes().stream()
                .filter(x -> "LLM_LORA".equals(x.key()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("LLM_LORA recipe not found in catalog"));
        // mobile_safe:false → desktop/mobile evaluateEligibility() hard-fails phones.
        assertEquals(Boolean.FALSE, r.requirements().mobileSafe());
        // min_ram_gb:8 → devices under 8GB hard-fail.
        assertEquals(8.0, r.requirements().minRamGb().doubleValue(), 0.001);
    }
}
