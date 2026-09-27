package com.federated.fl_platform_api.orchestration;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Terminates a spawned script together with everything it started.
 *
 * <p>The backend's scripts are bash wrappers that fork Python rather than exec it ({@code run_fl_server.sh} runs
 * {@code fl_server.py | tee}; the others run {@code "$PYTHON" script.py}). Killing only the wrapper orphans the real
 * work: an FL server keeps its port, a timed-out or stopped job keeps running. So the whole tree is killed. The
 * descendants are captured before anything is killed, because once the wrapper dies its children are re-parented
 * and no longer appear as its descendants.</p>
 */
public final class ProcessTrees {

    private ProcessTrees() {
    }

    /** Forcibly terminate {@code process} and every descendant it has at this moment. */
    public static void destroyForcibly(Process process) {
        process.descendants().toList().forEach(ProcessHandle::destroyForcibly);
        process.destroyForcibly();
    }

    /** Forcibly terminate {@code root} and every descendant; returns all the processes it signalled. */
    public static List<ProcessHandle> destroyForcibly(ProcessHandle root) {
        List<ProcessHandle> tree = new ArrayList<>(root.descendants().toList());
        tree.forEach(ProcessHandle::destroyForcibly);
        root.destroyForcibly();
        tree.add(root);
        return tree;
    }

    /** Wait up to {@code timeoutSeconds} for every process in {@code tree} to exit. */
    public static void awaitExit(List<ProcessHandle> tree, long timeoutSeconds)
            throws InterruptedException, ExecutionException, TimeoutException {
        CompletableFuture.allOf(tree.stream().map(ProcessHandle::onExit).toArray(CompletableFuture[]::new))
                .get(timeoutSeconds, TimeUnit.SECONDS);
    }
}
