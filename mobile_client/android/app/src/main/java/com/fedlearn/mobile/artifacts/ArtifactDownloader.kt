package com.fedlearn.mobile.artifacts

import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest

/** A download refused or failed for a named reason; [code] is what the app reports. */
class ArtifactDownloadException(val code: String, message: String) : IOException(message)

/**
 * Streams execution-contract artifacts into app-private storage (Stage 3, slice A1).
 *
 * Each artifact is declared by the contract with a SHA-256 and a byte size. The download is refused up front when
 * the device lacks room, streamed to a partial file while being hashed, aborted as soon as it exceeds the declared
 * size, and promoted atomically to `artifacts/<sha256>` only when size and hash both match. Anything else is moved to
 * `artifacts/quarantine/`. A partial file resumes with `Range` only when the server's validator from the first
 * attempt is held and still matches (`If-Range`); otherwise it restarts from zero.
 *
 * The OkHttp client is injected, so production can pass React Native's shared client and its session cookie jar.
 */
class ArtifactDownloader(
    private val client: OkHttpClient,
    root: File,
    private val usableSpace: () -> Long = { root.usableSpace },
) {
    data class Descriptor(val url: String, val sha256: String, val byteSize: Long)

    private val artifacts = File(root, "artifacts")
    private val partials = File(artifacts, "tmp")
    private val quarantine = File(artifacts, "quarantine")

    /** The verified local file for [d], downloading it if it is not already present. */
    fun fetch(d: Descriptor): File {
        if (!SHA256_HEX.matches(d.sha256) || d.byteSize < 0) {
            throw ArtifactDownloadException("ARTIFACT_INVALID_DESCRIPTOR", "not a sha256 and a size: ${d.sha256.take(80)}")
        }
        val target = File(artifacts, d.sha256)
        if (target.isFile && target.length() == d.byteSize && sha256(target) == d.sha256) {
            return target
        }
        partials.mkdirs()
        val part = File(partials, "${d.sha256}.part")
        val validatorFile = File(partials, "${d.sha256}.validator")
        val resumeFrom = if (part.isFile && validatorFile.isFile) part.length().coerceAtMost(d.byteSize) else 0L
        if (usableSpace() < d.byteSize - resumeFrom) {
            throw ArtifactDownloadException("INSUFFICIENT_STORAGE", "${d.byteSize} bytes needed")
        }

        val request = Request.Builder().url(d.url).apply {
            if (resumeFrom > 0) {
                header("Range", "bytes=$resumeFrom-")
                header("If-Range", validatorFile.readText())
            }
        }.build()
        client.newCall(request).execute().use { response ->
            val append = when (response.code) {
                206 -> resumeFrom > 0
                200 -> false
                else -> throw ArtifactDownloadException("ARTIFACT_HTTP_${response.code}", "download failed")
            }
            val validator = response.header("ETag")
            if (validator != null && !validator.startsWith("W/")) validatorFile.writeText(validator) else validatorFile.delete()

            val digest = MessageDigest.getInstance("SHA-256")
            var written = 0L
            if (append) {
                FileInputStream(part).use { input -> written = copyInto(input, digest, null, Long.MAX_VALUE) }
            } else {
                part.delete()
            }
            FileOutputStream(part, append).use { out ->
                val body = response.body ?: throw ArtifactDownloadException("ARTIFACT_HTTP_EMPTY", "no body")
                written += copyInto(body.byteStream(), digest, out, d.byteSize - written) { reject(part, validatorFile, d, it) }
            }
            if (written != d.byteSize) {
                reject(part, validatorFile, d, "ARTIFACT_SIZE_MISMATCH")
            }
            val actual = hex(digest.digest())
            if (actual != d.sha256) {
                reject(part, validatorFile, d, "ARTIFACT_HASH_MISMATCH")
            }
        }
        artifacts.mkdirs()
        Files.move(part.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        validatorFile.delete()
        return target
    }

    /** Delete quarantined files older than [retentionMillis]. */
    fun cleanQuarantine(nowMillis: Long = System.currentTimeMillis(), retentionMillis: Long = RETENTION_MILLIS) {
        quarantine.listFiles()?.forEach { if (nowMillis - it.lastModified() > retentionMillis) it.delete() }
    }

    /**
     * Copy [input] into [digest] (and [out] when given), at most [limit] bytes; one more byte calls [onOverflow],
     * which must throw.
     */
    private fun copyInto(
        input: java.io.InputStream,
        digest: MessageDigest,
        out: FileOutputStream?,
        limit: Long,
        onOverflow: (String) -> Nothing = { throw IllegalStateException(it) },
    ): Long {
        val buffer = ByteArray(64 * 1024)
        var total = 0L
        while (true) {
            val n = input.read(buffer)
            if (n < 0) return total
            if (total + n > limit) {
                onOverflow("ARTIFACT_OVERSIZE")
            }
            digest.update(buffer, 0, n)
            out?.write(buffer, 0, n)
            total += n
        }
    }

    private fun reject(part: File, validatorFile: File, d: Descriptor, code: String): Nothing {
        quarantine.mkdirs()
        if (part.exists()) {
            Files.move(part.toPath(), File(quarantine, "${d.sha256}-${System.nanoTime()}").toPath(),
                StandardCopyOption.REPLACE_EXISTING)
        }
        validatorFile.delete()
        throw ArtifactDownloadException(code, "${d.url}: $code")
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { copyInto(it, digest, null, Long.MAX_VALUE) }
        return hex(digest.digest())
    }

    private fun hex(bytes: ByteArray) = bytes.joinToString("") { "%02x".format(it) }

    companion object {
        private val SHA256_HEX = Regex("[0-9a-f]{64}")
        const val RETENTION_MILLIS = 24L * 60 * 60 * 1000
    }
}
