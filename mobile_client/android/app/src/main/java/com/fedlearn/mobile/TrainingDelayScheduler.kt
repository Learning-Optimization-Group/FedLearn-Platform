package com.fedlearn.mobile

import java.io.Closeable
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit

/** Round pacing that does not depend on activity frames or React Native timers. */
class TrainingDelayScheduler : Closeable {
    private class Wait(val complete: () -> Unit) {
        var future: ScheduledFuture<*>? = null
    }

    private val executor = ScheduledThreadPoolExecutor(1) { task ->
        Thread(task, "fedlearn-training-delay").apply { isDaemon = true }
    }.apply { removeOnCancelPolicy = true }
    private val pending = mutableSetOf<Wait>()
    private var closed = false

    fun wait(milliseconds: Double, complete: () -> Unit) {
        require(milliseconds.isFinite() && milliseconds >= 0 && milliseconds <= 60000 &&
            milliseconds == milliseconds.toLong().toDouble()) { "Invalid training delay" }
        synchronized(pending) {
            check(!closed) { "Training scheduler is closed" }
            check(pending.size < 8) { "Too many pending training waits" }
            val wait = Wait(complete)
            pending.add(wait)
            wait.future = executor.schedule({
                val deliver = synchronized(pending) { pending.remove(wait) }
                if (deliver) wait.complete()
            }, milliseconds.toLong(), TimeUnit.MILLISECONDS)
        }
    }

    /** Release a pacing wait promptly so the cooperative stop flag can be observed. */
    fun cancelPending() {
        val cancelled = synchronized(pending) {
            val waits = pending.toList()
            pending.clear()
            waits.forEach { it.future?.cancel(false) }
            waits
        }
        cancelled.forEach { it.complete() }
    }

    override fun close() {
        synchronized(pending) {
            closed = true
            pending.forEach { it.future?.cancel(false) }
            pending.clear()
            executor.shutdownNow()
        }
    }
}
