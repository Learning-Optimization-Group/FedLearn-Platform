package com.federated.fl_platform_api.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GenerationRequestValidationTest {
    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    private GenerationRequest req(String prompt, int max, double temp) {
        GenerationRequest r = new GenerationRequest();
        r.setPrompt(prompt); r.setMaxNewTokens(max); r.setTemperature(temp);
        return r;
    }

    @Test void validPasses()      { assertTrue(v.validate(req("hi", 256, 0.7)).isEmpty()); }
    @Test void blankPromptFails() { assertFalse(v.validate(req("  ", 256, 0.7)).isEmpty()); }
    @Test void maxTooHighFails()  { assertFalse(v.validate(req("hi", 4096, 0.7)).isEmpty()); }
    @Test void maxTooLowFails()   { assertFalse(v.validate(req("hi", 0, 0.7)).isEmpty()); }
    @Test void tempTooHighFails() { assertFalse(v.validate(req("hi", 256, 2.5)).isEmpty()); }
}
