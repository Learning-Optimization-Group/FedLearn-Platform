package com.federated.fl_platform_api.bootstrap;

import org.springframework.stereotype.Component;

import java.util.Optional;

/**
 * A thin, injectable seam over {@link ProcessHandle#of(long)} (BA-3). {@code ProcessHandle.of} is a
 * static method, so wrapping it lets {@link StartupReconciler}'s reconciliation logic be unit-tested
 * with mocked {@link ProcessHandle}s (each an interface, easily stubbed) instead of real OS processes.
 */
@Component
public class ProcessProbe {

    /** The live process for {@code pid}, or empty if no process with that id is currently alive. */
    public Optional<ProcessHandle> of(long pid) {
        return ProcessHandle.of(pid);
    }
}
