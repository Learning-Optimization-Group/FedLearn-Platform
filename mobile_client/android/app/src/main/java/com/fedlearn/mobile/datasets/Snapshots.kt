package com.fedlearn.mobile.datasets

import java.io.File
import java.io.IOException
import java.security.MessageDigest

/** An import or load refused for a named reason; [code] is what the app reports. */
class DatasetImportException(val code: String, message: String) : IOException(message)

/**
 * An immutable on-device dataset (Stage 3, section B), stored as `datasets/<snapshotId>/` in the layout the native
 * trainer reads: `inputs.f32` (row-major little-endian float32, recordCount × inputShape) and `targets.i64`
 * (little-endian int64 class indices), described by `snapshot.json`.
 *
 * [snapshotId] is the SHA-256 of the canonical JSON of every field except the id and the directory, so the same
 * content always has the same id.
 */
data class Snapshot(
    val snapshotId: String,
    val dir: File,
    val modality: String,
    val inputShape: List<Int>,
    val inputDtype: String,
    val classNames: List<String>,
    val labelSchemaId: String,
    val recordCount: Int,
    val inputsSha256: String,
    val targetsSha256: String,
) {
    internal fun contentFields(): Map<String, Any> = mapOf(
        "schemaVersion" to SCHEMA_VERSION,
        "modality" to modality,
        "inputShape" to inputShape,
        "inputDtype" to inputDtype,
        "classNames" to classNames,
        "labelSchemaId" to labelSchemaId,
        "recordCount" to recordCount,
        "inputsSha256" to inputsSha256,
        "targetsSha256" to targetsSha256,
    )

    companion object {
        const val SCHEMA_VERSION = 1

        internal fun idOf(content: Map<String, Any>): String = sha256Hex(CanonicalJson.encode(content).toByteArray())
    }
}

/** The execution contract's label-schema identifier, byte-identical to `execution_plan.label_schema_id` in Python. */
object LabelSchema {
    fun id(classNames: List<String>): String = "labels-sha256:" + sha256Hex(CanonicalJson.encode(classNames).toByteArray())
}

/**
 * JSON encoded exactly as Python's `json.dumps(value, sort_keys=True, separators=(",", ":"))` with its default
 * `ensure_ascii=True`, for the values these formats hold: strings, integers, lists and string-keyed maps. It is used
 * wherever a digest must match one computed in Python.
 */
object CanonicalJson {
    fun encode(value: Any?): String = StringBuilder().also { write(it, value) }.toString()

    private fun write(out: StringBuilder, value: Any?) {
        when (value) {
            null -> out.append("null")
            is String -> writeString(out, value)
            is Int, is Long -> out.append(value.toString())
            is Boolean -> out.append(if (value) "true" else "false")
            is List<*> -> {
                out.append('[')
                value.forEachIndexed { i, v -> if (i > 0) out.append(','); write(out, v) }
                out.append(']')
            }
            is Map<*, *> -> {
                out.append('{')
                value.entries.sortedBy { it.key as String }.forEachIndexed { i, (k, v) ->
                    if (i > 0) out.append(',')
                    writeString(out, k as String)
                    out.append(':')
                    write(out, v)
                }
                out.append('}')
            }
            else -> throw IllegalArgumentException("not encodable canonically: ${value::class.java.simpleName}")
        }
    }

    private fun writeString(out: StringBuilder, s: String) {
        out.append('"')
        for (c in s) {
            when (c) {
                '"' -> out.append("\\\"")
                '\\' -> out.append("\\\\")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                '\b' -> out.append("\\b")
                '\u000C' -> out.append("\\f")
                else -> if (c.code < 0x20 || c.code > 0x7e) out.append("\\u%04x".format(c.code)) else out.append(c)
            }
        }
        out.append('"')
    }
}

internal fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

internal fun sha256Hex(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}
