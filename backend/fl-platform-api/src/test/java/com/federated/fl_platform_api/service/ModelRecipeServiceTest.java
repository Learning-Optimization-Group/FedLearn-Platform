package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ModelRecipeDto;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * DA-10: recipes.py is the single source of truth for the model-recipe catalog. There is no
 * hard-coded Java fallback (it had already drifted — missing BLOOD_CNN/LLM_LORA), so a load failure
 * surfaces loudly instead of masking a broken catalog with a stale duplicate.
 */
class ModelRecipeServiceTest {

    /** Overrides the process-spawning describe step so we drive success/failure without spawning python. */
    static class StubService extends ModelRecipeService {
        private final Supplier<List<ModelRecipeDto>> describe;
        StubService(Supplier<List<ModelRecipeDto>> describe) { this.describe = describe; }
        @Override protected List<ModelRecipeDto> runDescribe() { return describe.get(); }
    }

    private static ModelRecipeDto recipe(String key) {
        return new ModelRecipeDto(key, key, "image", List.of(), List.of(), List.of(), null);
    }

    @Test
    void successfulDescribeIsReturnedAndCachedOnce() {
        AtomicInteger calls = new AtomicInteger();
        ModelRecipeService svc = new StubService(() -> {
            calls.incrementAndGet();
            return List.of(recipe("CNN"), recipe("LLM_LORA"));
        });
        assertEquals(2, svc.getRecipes().size());
        svc.getRecipes();                                    // served from cache
        assertEquals(1, calls.get(), "runDescribe must be called once, then cached");
    }

    @Test
    void failedDescribeThrows_withNoHardcodedFallback() {
        ModelRecipeService svc = new StubService(() -> { throw new RuntimeException("python missing"); });
        assertThrows(IllegalStateException.class, svc::getRecipes);
    }

    @Test
    void emptyDescribeThrows() {
        ModelRecipeService svc = new StubService(List::of);  // an empty catalog is a failure, not valid
        assertThrows(IllegalStateException.class, svc::getRecipes);
    }

    @Test
    void failedLoadIsNotCached_soItRetries() {
        AtomicInteger calls = new AtomicInteger();
        ModelRecipeService svc = new StubService(() -> {
            if (calls.getAndIncrement() == 0) throw new RuntimeException("transient");
            return List.of(recipe("CNN"));
        });
        assertThrows(IllegalStateException.class, svc::getRecipes);   // first attempt fails
        assertEquals(1, svc.getRecipes().size());                    // retry succeeds — failure not cached
    }
}
