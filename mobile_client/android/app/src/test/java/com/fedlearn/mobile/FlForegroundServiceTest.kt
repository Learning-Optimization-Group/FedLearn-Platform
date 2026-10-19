package com.fedlearn.mobile

import android.app.Application
import android.app.Service
import android.content.Intent
import android.os.Looper
import java.time.Duration
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.LooperMode

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class, manifest = Config.NONE, sdk = [29])
@LooperMode(LooperMode.Mode.PAUSED)
class FlForegroundServiceTest {
    @Test
    fun explicitStartShowsTrainingNotificationWithoutRequestingProcessRestart() {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            val result = service.onStartCommand(Intent(service, FlForegroundService::class.java), 0, 1)
            assertNotNull(shadowOf(service).lastForegroundNotification)
            assertFalse(shadowOf(service).isStoppedBySelf)
            assertEquals(Service.START_NOT_STICKY, result)
        } finally {
            controller.destroy()
        }
    }

    @Test
    fun emptyRestartStopsWithoutAdvertisingOrSamplingTraining() {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            assertEquals(Service.START_NOT_STICKY, service.onStartCommand(null, 0, 1))
            assertTrue(shadowOf(service).isStoppedBySelf)
            assertNull(shadowOf(service).lastForegroundNotification)
            // A stray sampler would attempt to load the unavailable device JNI library.
            shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(10))
        } finally {
            controller.destroy()
        }
    }

    @Test
    fun emptyRestartRemovesAnExistingNotificationAndPendingSampler() {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            service.onStartCommand(Intent(service, FlForegroundService::class.java), 0, 1)
            service.onStartCommand(null, 0, 2)
            assertTrue(shadowOf(service).isStoppedBySelf)
            assertNull(shadowOf(service).lastForegroundNotification)
            shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(10))
        } finally {
            controller.destroy()
        }
    }

    @Test
    fun explicitStopRemovesNotificationAndCancelsSampling() {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            service.onStartCommand(Intent(service, FlForegroundService::class.java), 0, 1)
            val result = service.onStartCommand(Intent().setAction(FlForegroundService.ACTION_STOP), 0, 2)
            assertEquals(Service.START_NOT_STICKY, result)
            assertTrue(shadowOf(service).isStoppedBySelf)
            assertNull(shadowOf(service).lastForegroundNotification)
            shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(10))
        } finally {
            controller.destroy()
        }
    }

    @Test
    fun retriedStartAfterProcessDeathDoesNotAdvertiseTraining() {
        assertSystemRestartStops(Service.START_FLAG_RETRY)
    }

    @Test
    fun redeliveredStartAfterProcessDeathDoesNotAdvertiseTraining() {
        assertSystemRestartStops(Service.START_FLAG_REDELIVERY)
    }

    @Test
    fun rejectedRestartOnlyStopsItsOwnStartRequest() {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            service.onStartCommand(Intent(service, FlForegroundService::class.java), Service.START_FLAG_RETRY, 42)
            assertEquals(42, shadowOf(service).stopSelfId)
        } finally {
            controller.destroy()
        }
    }

    private fun assertSystemRestartStops(flags: Int) {
        val controller = Robolectric.buildService(FlForegroundService::class.java).create()
        val service = controller.get()
        try {
            val result = service.onStartCommand(Intent(service, FlForegroundService::class.java), flags, 1)
            assertEquals(Service.START_NOT_STICKY, result)
            assertTrue(shadowOf(service).isStoppedBySelf)
            assertNull(shadowOf(service).lastForegroundNotification)
            shadowOf(Looper.getMainLooper()).idleFor(Duration.ofSeconds(10))
        } finally {
            controller.destroy()
        }
    }
}
