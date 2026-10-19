package com.federated.fl_platform_api.contract;

import java.util.List;

/**
 * What a run's staged on-device bundle says about itself (manifest.json): each model program's digest and size,
 * the runtime operators they require, the declared resource envelope, the trainable layout the programs load, and
 * the most examples one call of a program takes ({@code maxBatch}: the programs take 1..maxBatch).
 */
public record StagedBundle(List<ModelFile> modelFiles, List<String> requiredOperators, ResourceEnvelope envelope,
                           List<LayoutEntry> paramLayout, int maxBatch) {

    public StagedBundle {
        modelFiles = List.copyOf(modelFiles);
        requiredOperators = List.copyOf(requiredOperators);
        paramLayout = List.copyOf(paramLayout);
    }

    /** A staged model program, by its served file name. */
    public record ModelFile(String file, String sha256, long byteSize) {
    }

    public record ResourceEnvelope(long peakMemoryBytes, long storageBytes, long probeMs, long trainMs) {
    }

    public record LayoutEntry(String name, List<Long> shape) {

        public LayoutEntry {
            shape = List.copyOf(shape);
        }
    }
}
