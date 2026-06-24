package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.dto.DeviceRequirements;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class DeviceRequirementsTest {

    private DeviceRequirements recipe() {
        return new DeviceRequirements(8.0, 1.5, 27, "13.0", Boolean.FALSE,
                125_000_000L, 10.0, 30, 20, Boolean.FALSE, List.of("cpu"));
    }

    @Test
    void merge_overrideTightensRamAndStorage() {
        DeviceRequirements override = new DeviceRequirements(12.0, 2.0, null, null, null,
                null, null, null, null, null, null);
        DeviceRequirements eff = DeviceRequirements.merge(recipe(), override);
        assertEquals(12.0, eff.minRamGb());      // max(8,12)
        assertEquals(2.0, eff.minStorageGb());   // max(1.5,2.0)
    }

    @Test
    void merge_overrideCannotLoosenBelowRecipe() {
        DeviceRequirements override = new DeviceRequirements(4.0, 0.1, null, null, null,
                null, null, null, null, null, null);
        DeviceRequirements eff = DeviceRequirements.merge(recipe(), override);
        assertEquals(8.0, eff.minRamGb());       // max(8,4) -> recipe wins
        assertEquals(1.5, eff.minStorageGb());
    }

    @Test
    void merge_mobileSafeIsAndedAndWifiIsOred() {
        // recipe mobileSafe=FALSE, requiresWifi=FALSE; override mobileSafe=TRUE, requiresWifi=TRUE
        DeviceRequirements override = new DeviceRequirements(null, null, null, null, Boolean.TRUE,
                null, null, null, null, Boolean.TRUE, null);
        DeviceRequirements eff = DeviceRequirements.merge(recipe(), override);
        assertEquals(Boolean.FALSE, eff.mobileSafe());  // FALSE if either FALSE
        assertEquals(Boolean.TRUE, eff.requiresWifi()); // OR
    }

    @Test
    void merge_iosVersionTakesStricter() {
        DeviceRequirements override = new DeviceRequirements(null, null, null, "16.0", null,
                null, null, null, null, null, null);
        assertEquals("16.0", DeviceRequirements.merge(recipe(), override).minOsIos()); // 16.0 > 13.0
        DeviceRequirements looser = new DeviceRequirements(null, null, null, "12.0", null,
                null, null, null, null, null, null);
        assertEquals("13.0", DeviceRequirements.merge(recipe(), looser).minOsIos());   // recipe wins
    }

    @Test
    void merge_recipeOnlyFieldsIgnoreOverride() {
        DeviceRequirements override = new DeviceRequirements(null, null, null, null, null,
                999L, 999.0, null, null, null, List.of("nnapi"));
        DeviceRequirements eff = DeviceRequirements.merge(recipe(), override);
        assertEquals(125_000_000L, eff.maxTrainableParams());          // recipe-only
        assertEquals(30, eff.estimatedRoundTimeSeconds());             // recipe-only
        assertEquals(List.of("cpu"), eff.acceleratorBackends());       // recipe-only
    }

    @Test
    void merge_nullOverrideReturnsRecipe() {
        assertEquals(recipe(), DeviceRequirements.merge(recipe(), null));
    }

    @Test
    void merge_batteryPctTakesMax() {
        // recipe minBatteryPct=20, override minBatteryPct=40 → effective 40 (most-restrictive = max)
        DeviceRequirements override = new DeviceRequirements(null, null, null, null, null,
                null, null, null, 40, null, null);
        DeviceRequirements eff = DeviceRequirements.merge(recipe(), override);
        assertEquals(40, eff.minBatteryPct()); // max(20, 40)
    }
}
