package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FedOptSettingsTest {
    @Test void omittedFieldsResolveHistoricalDefaults() {
        var settings = new FedOptSettings(null, null, null, null, null);
        assertEquals("adam", settings.variant());
        assertEquals(1., settings.serverLearningRate());
        assertEquals(.9, settings.beta1());
        assertEquals(.99, settings.beta2());
        assertEquals(.001, settings.tau());
    }
    @Test void settingsRejectNonFiniteAndInvalidValues() {
        for (double rate : new double[]{0, -1, 11, Double.NaN, Double.POSITIVE_INFINITY}) {
            assertThrows(IllegalArgumentException.class, () -> new FedOptSettings("adam", rate, null, null, null));
        }
        assertThrows(IllegalArgumentException.class, () -> new FedOptSettings("bogus", null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new FedOptSettings("adam", null, 1., null, null));
        assertThrows(IllegalArgumentException.class, () -> new FedOptSettings("adam", null, null, -.1, null));
        assertThrows(IllegalArgumentException.class, () -> new FedOptSettings("adam", null, null, null, 0.));
    }
    @Test void runRoundTripsResolvedSettings() {
        var run = new Run();
        var settings = new FedOptSettings("yogi", .01, .8, .95, .002);
        run.setFedOpt(settings);
        assertEquals(settings, run.getFedOpt());
        run.setFedOpt(null);
        assertNull(run.getFedOpt());
    }
}
