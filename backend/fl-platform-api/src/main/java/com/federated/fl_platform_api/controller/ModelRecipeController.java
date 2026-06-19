package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import com.federated.fl_platform_api.service.ModelRecipeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * Exposes the model-recipe catalog to the frontend so the model picker, input
 * collection, and label rendering can be driven from one source of truth.
 * Authenticated by default (not in {@code SecurityConfig} public paths), same
 * posture as {@link InferenceController}.
 */
@RestController
@RequestMapping("/api/model-recipes")
public class ModelRecipeController {

    private final ModelRecipeService modelRecipeService;

    public ModelRecipeController(ModelRecipeService modelRecipeService) {
        this.modelRecipeService = modelRecipeService;
    }

    /** The full model-recipe catalog (input kind, class labels, base models, optimizers). */
    @GetMapping
    public ResponseEntity<List<ModelRecipeDto>> listRecipes() {
        return ResponseEntity.ok(modelRecipeService.getRecipes());
    }
}
