package com.fedlearn.mobile.artifacts

import android.app.Application
import android.webkit.CookieManager
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.BridgeReactContext
import java.lang.reflect.Proxy
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.Buffer
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class, manifest = Config.NONE, sdk = [29])
class ArtifactCredentialIsolationTest {
    private fun download(url: String, origin: String, token: String, payload: ByteArray): Any? {
        val digest = MessageDigest.getInstance("SHA-256").digest(payload).joinToString("") { "%02x".format(it) }
        val completed = CompletableFuture<Any?>()
        val promise = Proxy.newProxyInstance(Promise::class.java.classLoader, arrayOf(Promise::class.java)) { _, method, args ->
            when (method.name) {
                "resolve" -> completed.complete(args?.get(0))
                "reject" -> completed.completeExceptionally(AssertionError("Native download rejected: " + args?.get(0)))
            }
            null
        } as Promise
        val context = BridgeReactContext(RuntimeEnvironment.getApplication())
        ArtifactServiceModule(context).fetchArtifact(url, digest, payload.size.toDouble(), origin, token, promise)
        return completed.get(15, TimeUnit.SECONDS)
    }

    @Test
    fun artifactDownloadDoesNotForwardTheSharedLoginCookieToAnotherPort() {
        val issuer = MockWebServer()
        val receiver = MockWebServer()
        issuer.start()
        receiver.start()
        try {
            val cookies = CookieManager.getInstance()
            cookies.removeAllCookies(null)
            cookies.setCookie(issuer.url("/").toString(), "jwtToken=placeholder-cookie-token; Path=/")
            cookies.flush()
            assertTrue(cookies.getCookie(receiver.url("/artifact").toString()).contains("placeholder-cookie-token"))
            val payload = ("cookie-isolation-control-" + UUID.randomUUID()).toByteArray()
            val digest = MessageDigest.getInstance("SHA-256").digest(payload).joinToString("") { "%02x".format(it) }
            receiver.enqueue(MockResponse().setBody(Buffer().write(payload)))
            val completed = CompletableFuture<Any?>()
            val promise = Proxy.newProxyInstance(Promise::class.java.classLoader, arrayOf(Promise::class.java)) { _, method, args ->
                when (method.name) {
                    "resolve" -> completed.complete(args?.get(0))
                    "reject" -> completed.completeExceptionally(AssertionError("Native download rejected: " + args?.get(0)))
                }
                null
            } as Promise
            val context = BridgeReactContext(RuntimeEnvironment.getApplication())
            ArtifactServiceModule(context).fetchArtifact(receiver.url("/artifact").toString(), digest, payload.size.toDouble(),
                receiver.url("/").toString(), "", promise)
            completed.get(15, TimeUnit.SECONDS)
            val request = receiver.takeRequest(5, TimeUnit.SECONDS)
            assertNotNull(request)
            assertNull("An origin-bound download must not carry the shared login cookie", request!!.getHeader("Cookie"))
        } finally {
            CookieManager.getInstance().removeAllCookies(null)
            receiver.shutdown()
            issuer.shutdown()
        }
    }

    @Test
    fun authenticatedDownloadUsesItsCapturedBearerAndNoAmbientCookie() {
        val server = MockWebServer()
        server.start()
        try {
            CookieManager.getInstance().setCookie(server.url("/").toString(), "jwtToken=placeholder-cookie-token; Path=/")
            val payload = ("bound-download-" + UUID.randomUUID()).toByteArray()
            server.enqueue(MockResponse().setBody(Buffer().write(payload)))
            val path = download(server.url("/artifact").toString(), server.url("/").toString(), "placeholder-bound-token", payload) as String
            assertArrayEquals(payload, java.io.File(path).readBytes())
            val request = server.takeRequest(5, TimeUnit.SECONDS)!!
            assertEquals("Bearer placeholder-bound-token", request.getHeader("Authorization"))
            assertEquals("fedlearn-mobile", request.getHeader("X-FedLearn-Client"))
            assertNull(request.getHeader("Cookie"))
        } finally {
            CookieManager.getInstance().removeAllCookies(null)
            server.shutdown()
        }
    }

    @Test
    fun aCredentialForAnotherPortIsRefusedBeforeSendingARequest() {
        val issuer = MockWebServer()
        val receiver = MockWebServer()
        issuer.start()
        receiver.start()
        try {
            val payload = ("foreign-download-" + UUID.randomUUID()).toByteArray()
            receiver.enqueue(MockResponse().setBody(Buffer().write(payload)))
            val error = assertThrows(ExecutionException::class.java) {
                download(receiver.url("/artifact").toString(), issuer.url("/").toString(), "placeholder-bound-token", payload)
            }
            assertTrue(error.cause!!.message!!.contains("ARTIFACT_AUTH_SCOPE_MISMATCH"))
            assertEquals(0, receiver.requestCount)
            assertEquals(0, issuer.requestCount)
        } finally {
            receiver.shutdown()
            issuer.shutdown()
        }
    }

    @Test
    fun redirectCannotForwardTheCapturedCredentialToAnotherPort() {
        val issuer = MockWebServer()
        val receiver = MockWebServer()
        issuer.start()
        receiver.start()
        try {
            val payload = ("redirect-download-" + UUID.randomUUID()).toByteArray()
            issuer.enqueue(MockResponse().setResponseCode(302).addHeader("Location", receiver.url("/artifact").toString()))
            receiver.enqueue(MockResponse().setBody(Buffer().write(payload)))
            val error = assertThrows(ExecutionException::class.java) {
                download(issuer.url("/artifact").toString(), issuer.url("/").toString(), "placeholder-bound-token", payload)
            }
            assertTrue(error.cause!!.message!!.contains("ARTIFACT_HTTP_302"))
            assertEquals("Bearer placeholder-bound-token", issuer.takeRequest(5, TimeUnit.SECONDS)!!.getHeader("Authorization"))
            assertEquals(0, receiver.requestCount)
        } finally {
            receiver.shutdown()
            issuer.shutdown()
        }
    }
}
