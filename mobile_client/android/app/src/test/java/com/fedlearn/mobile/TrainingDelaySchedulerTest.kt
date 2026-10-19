package com.fedlearn.mobile

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class TrainingDelaySchedulerTest {
    @Test
    fun waitCompletesWithoutActivityFrames() {
        val scheduler = TrainingDelayScheduler()
        try {
            val done = CountDownLatch(1)
            scheduler.wait(10.0) { done.countDown() }
            assertTrue(done.await(1, TimeUnit.SECONDS))
        } finally {
            scheduler.close()
        }
    }

    @Test
    fun stopReleasesPendingWaitOnceAndAllowsAnotherRun() {
        val scheduler = TrainingDelayScheduler()
        try {
            val completions = AtomicInteger()
            scheduler.wait(60000.0) { completions.incrementAndGet() }
            scheduler.cancelPending()
            scheduler.cancelPending()
            assertEquals(1, completions.get())
            val restarted = CountDownLatch(1)
            scheduler.wait(0.0) { restarted.countDown() }
            assertTrue(restarted.await(1, TimeUnit.SECONDS))
        } finally {
            scheduler.close()
        }
    }

    @Test
    fun rejectsMalformedOrUnboundedWaits() {
        val scheduler = TrainingDelayScheduler()
        try {
            for (delay in listOf(-1.0, Double.NaN, Double.POSITIVE_INFINITY, 0.5, 60001.0)) {
                try {
                    scheduler.wait(delay) { fail("Rejected wait completed") }
                    fail("Invalid wait accepted")
                } catch (_: IllegalArgumentException) {
                }
            }
        } finally {
            scheduler.close()
        }
    }

    @Test
    fun boundsPendingWaitsAndClosingCancelsWithoutLateCallbacks() {
        val scheduler = TrainingDelayScheduler()
        val completions = AtomicInteger()
        repeat(8) { scheduler.wait(60000.0) { completions.incrementAndGet() } }
        try {
            scheduler.wait(60000.0) { fail("Overflow completed") }
            fail("Unbounded pending waits accepted")
        } catch (_: IllegalStateException) {
        }
        scheduler.close()
        assertEquals(0, completions.get())
        try {
            scheduler.wait(0.0) { fail("Closed scheduler completed") }
            fail("Closed scheduler accepted a wait")
        } catch (_: IllegalStateException) {
        }
    }
}
