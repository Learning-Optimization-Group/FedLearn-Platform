package com.fedlearn.mobile.artifacts

import okhttp3.OkHttpClient
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Stage 3 slice A1: artifacts stream to app-private files, bounded by their declared size and verified by their
 * declared SHA-256 before they are promoted, instead of travelling base64-encoded through the JavaScript bridge.
 */
class ArtifactDownloaderTest {

    @get:Rule val tmp = TemporaryFolder()

    private lateinit var server: MockWebServer
    private val requests = CopyOnWriteArrayList<Map<String, String?>>()
    private var body: ByteArray = ByteArray(0)
    private var etag: String = "\"v1\""
    private var honourRange = true

    private val payload = ByteArray(200_000) { (it * 31 % 251).toByte() }
    private val payloadSha = sha256(payload)

    @Before
    fun startServer() {
        body = payload
        server = MockWebServer()
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = serve(request)
        }
        server.start()
    }

    @After
    fun stopServer() = server.shutdown()

    private fun serve(request: RecordedRequest): MockResponse {
        val range = request.getHeader("Range")
        val ifRange = request.getHeader("If-Range")
        requests += mapOf("Range" to range, "If-Range" to ifRange)
        if (range != null && honourRange && ifRange == etag) {
            val start = range.removePrefix("bytes=").removeSuffix("-").toInt()
            val part = body.copyOfRange(start, body.size)
            return MockResponse().setResponseCode(206).addHeader("ETag", etag)
                .addHeader("Content-Range", "bytes $start-${body.size - 1}/${body.size}")
                .setBody(Buffer().write(part))
        }
        return MockResponse().setResponseCode(200).addHeader("ETag", etag).setBody(Buffer().write(body))
    }

    private fun url() = server.url("/a").toString()

    private fun downloader(space: Long = Long.MAX_VALUE) =
        ArtifactDownloader(OkHttpClient(), tmp.root, usableSpace = { space })

    private fun descriptor(sha: String = payloadSha, size: Long = payload.size.toLong()) =
        ArtifactDownloader.Descriptor(url(), sha, size)

    private fun expectFailure(code: String, block: () -> Unit): ArtifactDownloadException {
        try {
            block()
        } catch (e: ArtifactDownloadException) {
            assertEquals(code, e.code)
            return e
        }
        fail("expected ArtifactDownloadException $code")
        throw AssertionError()
    }

    @Test
    fun downloadsVerifiesAndPromotesToAContentAddressedFile() {
        val file = downloader().fetch(descriptor())

        assertEquals(File(tmp.root, "artifacts/$payloadSha"), file)
        assertArrayEquals(payload, file.readBytes())
        assertFalse("no partial file is left behind", File(tmp.root, "artifacts/tmp/$payloadSha.part").exists())
    }

    @Test
    fun aVerifiedArtifactAlreadyOnDiskNeedsNoRequest() {
        downloader().fetch(descriptor())
        requests.clear()

        downloader().fetch(descriptor())

        assertTrue(requests.isEmpty())
    }

    @Test
    fun aHashMismatchIsQuarantinedAndNeverPromoted() {
        body = payload.copyOf().also { it[10] = (it[10] + 1).toByte() }

        expectFailure("ARTIFACT_HASH_MISMATCH") { downloader().fetch(descriptor()) }

        assertFalse(File(tmp.root, "artifacts/$payloadSha").exists())
        assertEquals(1, File(tmp.root, "artifacts/quarantine").listFiles()!!.size)
    }

    @Test
    fun aResponseLargerThanDeclaredIsAbortedAndQuarantined() {
        body = payload + ByteArray(10)

        expectFailure("ARTIFACT_OVERSIZE") { downloader().fetch(descriptor()) }

        assertFalse(File(tmp.root, "artifacts/$payloadSha").exists())
    }

    @Test
    fun aResponseShorterThanDeclaredIsRejected() {
        body = payload.copyOf(payload.size - 1)

        expectFailure("ARTIFACT_SIZE_MISMATCH") { downloader().fetch(descriptor()) }

        assertFalse(File(tmp.root, "artifacts/$payloadSha").exists())
    }

    @Test
    fun insufficientStorageIsRefusedBeforeAnyRequest() {
        expectFailure("INSUFFICIENT_STORAGE") { downloader(space = 1000).fetch(descriptor()) }

        assertTrue(requests.isEmpty())
    }

    @Test
    fun aPartialDownloadResumesWithRangeWhenTheValidatorStillMatches() {
        seedPartial(half = payload.size / 2, validator = etag)

        val file = downloader().fetch(descriptor())

        assertArrayEquals(payload, file.readBytes())
        assertEquals("bytes=${payload.size / 2}-", requests.single()["Range"])
        assertEquals(etag, requests.single()["If-Range"])
    }

    @Test
    fun aPartialDownloadRestartsWhenTheServerNoLongerHonoursIt() {
        seedPartial(half = payload.size / 2, validator = "\"stale\"")

        val file = downloader().fetch(descriptor())

        assertArrayEquals(payload, file.readBytes())
    }

    @Test
    fun aPartialWithoutAValidatorIsNotResumed() {
        seedPartial(half = payload.size / 2, validator = null)

        downloader().fetch(descriptor())

        assertEquals(null, requests.single()["Range"])
    }

    @Test
    fun aHashThatIsNotSha256HexIsRejectedBeforeItCanNameAPath() {
        expectFailure("ARTIFACT_INVALID_DESCRIPTOR") { downloader().fetch(descriptor(sha = "../../etc/passwd")) }
        expectFailure("ARTIFACT_INVALID_DESCRIPTOR") { downloader().fetch(descriptor(size = -1)) }
        assertTrue(requests.isEmpty())
    }

    @Test
    fun quarantineOlderThanTheRetentionIsCleanedAndNewerIsKept() {
        val q = File(tmp.root, "artifacts/quarantine").apply { mkdirs() }
        val old = File(q, "old").apply { writeText("x"); setLastModified(1_000L) }
        val fresh = File(q, "fresh").apply { writeText("x"); setLastModified(90_000_000L) }

        downloader().cleanQuarantine(nowMillis = 90_000_000L + 1)

        assertFalse(old.exists())
        assertTrue(fresh.exists())
    }

    private fun seedPartial(half: Int, validator: String?) {
        val dir = File(tmp.root, "artifacts/tmp").apply { mkdirs() }
        File(dir, "$payloadSha.part").writeBytes(payload.copyOf(half))
        if (validator != null) File(dir, "$payloadSha.validator").writeText(validator)
    }

    private fun sha256(bytes: ByteArray) =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}
