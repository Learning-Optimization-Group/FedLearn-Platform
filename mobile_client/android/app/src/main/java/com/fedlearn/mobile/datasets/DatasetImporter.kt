package com.fedlearn.mobile.datasets

import com.google.gson.JsonElement
import com.google.gson.JsonParseException
import com.google.gson.JsonParser
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.File
import java.io.FilterInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.DigestOutputStream
import java.security.MessageDigest
import java.util.Base64
import java.util.UUID

/**
 * Turns a user's vector data into an immutable [Snapshot] (Stage 3 slice B1), or refuses it whole with a named reason.
 *
 * Two sources: a CSV (a header naming a `label` column plus exactly `inputWidth` feature columns; unquoted fields;
 * labels are class names), and the canonical package (`dataset.json` + `records.jsonl`). Both are written straight to
 * the native layout in a temporary directory, hashed while written, and promoted atomically to
 * `datasets/<snapshotId>/`. A refused import leaves nothing behind. Class names are the run's (the recipe's classes,
 * checked against the contract's label schema before any import), and every label must be one of them.
 */
class DatasetImporter(
    root: File,
    private val usableSpace: () -> Long = { root.usableSpace },
    private val limits: Limits = Limits(),
) {
    data class Limits(
        val maxRecords: Int = 1_000_000,
        val maxSourceBytes: Long = 512L * 1024 * 1024,
        val minFreeBytes: Long = 16L * 1024 * 1024,
    )

    private val datasets = File(root, "datasets")

    fun importCsv(source: InputStream, classNames: List<String>, inputWidth: Int): Snapshot =
        build(classNames, inputWidth) { writer ->
            val reader = utf8Reader(Bounded(source, limits.maxSourceBytes))
            val header = readLineOrNull(reader)?.split(',')?.map { it.trim() }
                ?: throw DatasetImportException("DATASET_EMPTY", "no header")
            val labelAt = header.indexOf("label")
            if (labelAt < 0 || header.count { it == "label" } != 1) {
                throw DatasetImportException("DATASET_BAD_HEADER", "the header needs exactly one 'label' column")
            }
            if (header.size - 1 != inputWidth) {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH",
                    "${header.size - 1} feature columns; this run's model takes $inputWidth")
            }
            val values = FloatArray(inputWidth)
            while (true) {
                val line = readLineOrNull(reader) ?: break
                if (line.isBlank()) continue
                if (line.contains('"')) throw DatasetImportException("DATASET_BAD_ROW", "quoted fields are not supported")
                val fields = line.split(',')
                if (fields.size != header.size) {
                    throw DatasetImportException("DATASET_BAD_ROW", "row ${writer.count + 1} has ${fields.size} columns")
                }
                var v = 0
                for ((i, field) in fields.withIndex()) {
                    if (i != labelAt) values[v++] = parseValue(field.trim())
                }
                writer.add(values, labelIndex(fields[labelAt].trim(), classNames))
            }
        }

    fun importPackage(datasetJson: InputStream, recordsJsonl: InputStream, classNames: List<String>, inputWidth: Int): Snapshot =
        build(classNames, inputWidth) { writer ->
            val meta = try {
                JsonParser.parseReader(utf8Reader(Bounded(datasetJson, limits.maxSourceBytes))).asJsonObject
            } catch (e: CharacterCodingException) {
                throw DatasetImportException("DATASET_NOT_UTF8", "dataset.json is not UTF-8")
            } catch (e: RuntimeException) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "dataset.json is not a JSON object")
            }
            if (meta.get("schemaVersion")?.asIntOrNull() != 1) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "unsupported package schemaVersion")
            }
            if (meta.get("modality")?.asStringOrNull() != "vector") {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "only vector packages are supported")
            }
            val shape = meta.get("inputShape")?.takeIf { it.isJsonArray }?.asJsonArray?.map { it.asIntOrNull() }
            if (shape != listOf(inputWidth)) {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH", "inputShape $shape; this run's model takes [$inputWidth]")
            }
            val declaredClasses = meta.get("classNames")?.takeIf { it.isJsonArray }?.asJsonArray?.map { it.asStringOrNull() }
            if (declaredClasses != classNames) {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH", "the package's classes are not this run's")
            }
            val declaredCount = meta.get("recordCount")?.asIntOrNull()
                ?: throw DatasetImportException("DATASET_BAD_PACKAGE", "recordCount is missing")

            val reader = utf8Reader(Bounded(recordsJsonl, limits.maxSourceBytes))
            val values = FloatArray(inputWidth)
            while (true) {
                val line = readLineOrNull(reader) ?: break
                if (line.isBlank()) continue
                val record = try {
                    JsonParser.parseString(line).asJsonObject
                } catch (e: RuntimeException) {
                    throw DatasetImportException("DATASET_BAD_ROW", "record ${writer.count + 1} is not a JSON object")
                }
                val label = record.get("label")?.asStringOrNull()
                    ?: throw DatasetImportException("DATASET_BAD_ROW", "record ${writer.count + 1} has no label")
                val array = record.get("values")?.takeIf { it.isJsonArray }?.asJsonArray
                if (array == null || array.size() != inputWidth) {
                    throw DatasetImportException("DATASET_BAD_ROW", "record ${writer.count + 1} needs $inputWidth values")
                }
                for (i in 0 until inputWidth) {
                    val e = array[i]
                    values[i] = if (e.isJsonPrimitive && e.asJsonPrimitive.isNumber) checkFinite(e.asFloat) else parseValue(e.toString().trim('"'))
                }
                writer.add(values, labelIndex(label, classNames))
            }
            if (writer.count != declaredCount) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "dataset.json declares $declaredCount records; found ${writer.count}")
            }
        }

    /**
     * An image package (Stage 4 S5): `dataset.json` states `modality` "image", the `height`, `width` and `channels`,
     * the classes and the record count; each `records.jsonl` line is `{"label": ..., "pixels": base64}`, the pixels an
     * 8-bit HWC image. Each image is prepared by the run's transforms ([shape]) as it is written, so the snapshot holds
     * exactly the tensors the model takes.
     */
    fun importImagePackage(datasetJson: InputStream, recordsJsonl: InputStream, classNames: List<String>,
                           shape: DataShape.Image): Snapshot =
        build(classNames, shape) { writer ->
            val meta = try {
                JsonParser.parseReader(utf8Reader(Bounded(datasetJson, limits.maxSourceBytes))).asJsonObject
            } catch (e: CharacterCodingException) {
                throw DatasetImportException("DATASET_NOT_UTF8", "dataset.json is not UTF-8")
            } catch (e: RuntimeException) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "dataset.json is not a JSON object")
            }
            if (meta.get("schemaVersion")?.asIntOrNull() != 1) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "unsupported package schemaVersion")
            }
            if (meta.get("modality")?.asStringOrNull() != "image") {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH", "this run's model takes images")
            }
            val stated = listOf("height", "width", "channels").map { meta.get(it)?.asIntOrNull() }
            if (stated != listOf(shape.height, shape.width, shape.channels)) {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH",
                    "images of ${stated.joinToString("x")}; this run's model takes " +
                        "${shape.height}x${shape.width}x${shape.channels}")
            }
            val declaredClasses = meta.get("classNames")?.takeIf { it.isJsonArray }?.asJsonArray?.map { it.asStringOrNull() }
            if (declaredClasses != classNames) {
                throw DatasetImportException("DATASET_SCHEMA_MISMATCH", "the package's classes are not this run's")
            }
            val declaredCount = meta.get("recordCount")?.asIntOrNull()
                ?: throw DatasetImportException("DATASET_BAD_PACKAGE", "recordCount is missing")
            val reader = utf8Reader(Bounded(recordsJsonl, limits.maxSourceBytes))
            val decoder = Base64.getDecoder()
            while (true) {
                val line = readLineOrNull(reader) ?: break
                if (line.isBlank()) continue
                val n = writer.count + 1
                val record = try {
                    JsonParser.parseString(line).asJsonObject
                } catch (e: RuntimeException) {
                    throw DatasetImportException("DATASET_BAD_ROW", "record $n is not a JSON object")
                }
                val label = record.get("label")?.asStringOrNull()
                    ?: throw DatasetImportException("DATASET_BAD_ROW", "record $n has no label")
                val encoded = record.get("pixels")?.asStringOrNull()
                    ?: throw DatasetImportException("DATASET_BAD_ROW", "record $n has no pixels")
                val pixels = try {
                    decoder.decode(encoded)
                } catch (e: IllegalArgumentException) {
                    throw DatasetImportException("DATASET_BAD_ROW", "record $n's pixels are not base64")
                }
                if (pixels.size != shape.elementCount) {
                    throw DatasetImportException("DATASET_BAD_ROW",
                        "record $n has ${pixels.size} pixel values; the images are ${shape.elementCount}")
                }
                writer.add(ImageTransforms.toTensor(pixels, shape), labelIndex(label, classNames))
            }
            if (writer.count != declaredCount) {
                throw DatasetImportException("DATASET_BAD_PACKAGE", "dataset.json declares $declaredCount records; found ${writer.count}")
            }
        }

    private fun build(classNames: List<String>, inputWidth: Int, fill: (Writer) -> Unit): Snapshot =
        build(classNames, DataShape.Vector(inputWidth), fill)

    private fun build(classNames: List<String>, shape: DataShape, fill: (Writer) -> Unit): Snapshot {
        require(shape.elementCount > 0 && classNames.isNotEmpty())
        if (usableSpace() < limits.minFreeBytes) {
            throw DatasetImportException("INSUFFICIENT_STORAGE", "not enough free storage to import")
        }
        datasets.mkdirs()
        val tmp = File(datasets, "tmp-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            val writer = Writer(tmp, shape.elementCount)
            writer.use { fill(it) }
            if (writer.count == 0) throw DatasetImportException("DATASET_EMPTY", "no records")
            val draft = when (shape) {
                is DataShape.Vector -> Snapshot("", tmp, "vector", listOf(shape.width), "f32", classNames,
                    LabelSchema.id(classNames), writer.count, writer.inputsSha256(), writer.targetsSha256())
                is DataShape.Image -> Snapshot("", tmp, "image", listOf(shape.channels, shape.height, shape.width),
                    "f32", classNames, LabelSchema.id(classNames), writer.count, writer.inputsSha256(),
                    writer.targetsSha256(), transforms = shape.transformsId)
            }
            val id = Snapshot.idOf(draft.contentFields())
            val target = File(datasets, id)
            val snapshot = draft.copy(snapshotId = id, dir = target)
            File(tmp, "snapshot.json").writeText(CanonicalJson.encode(snapshot.contentFields() + ("snapshotId" to id)))
            if (target.exists()) {
                tmp.deleteRecursively()   // the same content is already a snapshot
            } else {
                Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
            }
            return snapshot
        } catch (e: Throwable) {
            tmp.deleteRecursively()
            throw e
        }
    }

    /** Streams records into inputs.f32 / targets.i64, hashing as it writes. */
    private inner class Writer(dir: File, private val width: Int) : AutoCloseable {
        private val inputsDigest = MessageDigest.getInstance("SHA-256")
        private val targetsDigest = MessageDigest.getInstance("SHA-256")
        private val inputs: OutputStream =
            DigestOutputStream(BufferedOutputStream(File(dir, "inputs.f32").outputStream()), inputsDigest)
        private val targets: OutputStream =
            DigestOutputStream(BufferedOutputStream(File(dir, "targets.i64").outputStream()), targetsDigest)
        private val row = ByteBuffer.allocate(width * 4).order(ByteOrder.LITTLE_ENDIAN)
        private val label = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        private var bytes = 0L
        var count = 0
            private set

        fun add(values: FloatArray, labelIndex: Int) {
            if (count >= limits.maxRecords) {
                throw DatasetImportException("DATASET_TOO_MANY_RECORDS", "more than ${limits.maxRecords} records")
            }
            row.clear(); values.forEach { row.putFloat(it) }
            inputs.write(row.array())
            label.clear(); label.putLong(labelIndex.toLong())
            targets.write(label.array())
            count++
            bytes += width * 4L + 8
            if (count % 4096 == 0 && usableSpace() < limits.minFreeBytes) {
                throw DatasetImportException("INSUFFICIENT_STORAGE", "storage ran out during the import")
            }
        }

        fun inputsSha256(): String = hex(inputsDigest.digest())
        fun targetsSha256(): String = hex(targetsDigest.digest())

        override fun close() {
            inputs.close()
            targets.close()
        }

        private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
    }

    /** Counts source bytes and refuses the import past the limit. */
    private class Bounded(input: InputStream, private val limit: Long) : FilterInputStream(input) {
        private var seen = 0L

        override fun read(): Int = super.read().also { if (it >= 0) count(1) }

        override fun read(b: ByteArray, off: Int, len: Int): Int = super.read(b, off, len).also { if (it > 0) count(it) }

        private fun count(n: Int) {
            seen += n
            if (seen > limit) throw DatasetImportException("DATASET_TOO_LARGE", "the source is larger than $limit bytes")
        }
    }

    private fun utf8Reader(input: InputStream) = BufferedReader(InputStreamReader(input, Charsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)))

    private fun readLineOrNull(reader: BufferedReader): String? = try {
        reader.readLine()
    } catch (e: CharacterCodingException) {
        throw DatasetImportException("DATASET_NOT_UTF8", "the source is not UTF-8")
    }

    private fun labelIndex(label: String, classNames: List<String>): Int {
        val i = classNames.indexOf(label)
        if (i < 0) throw DatasetImportException("DATASET_UNKNOWN_LABEL", "'${label.take(64)}' is not one of this run's classes")
        return i
    }

    private fun parseValue(text: String): Float {
        if (!DECIMAL.matches(text)) {
            if (NON_FINITE.matches(text)) throw DatasetImportException("DATASET_NON_FINITE", "'$text' is not a finite number")
            throw DatasetImportException("DATASET_BAD_VALUE", "'${text.take(32)}' is not a decimal number")
        }
        return checkFinite(text.toFloat())
    }

    private fun checkFinite(v: Float): Float {
        if (!v.isFinite()) throw DatasetImportException("DATASET_NON_FINITE", "a value is not finite in float32")
        return v
    }

    private fun JsonElement.asIntOrNull(): Int? =
        if (isJsonPrimitive && asJsonPrimitive.isNumber) asNumber.toDouble().let { d -> d.toInt().takeIf { it.toDouble() == d } } else null

    private fun JsonElement.asStringOrNull(): String? =
        if (isJsonPrimitive && asJsonPrimitive.isString) asString else null

    companion object {
        private val DECIMAL = Regex("[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?")
        private val NON_FINITE = Regex("(?i)[+-]?(nan|inf|infinity)")
    }
}

/** Lists, loads (verifying) and deletes the device's snapshots. */
class DatasetStore(root: File) {
    private val datasets = File(root, "datasets")

    fun list(): List<Snapshot> = (datasets.listFiles() ?: emptyArray())
        .filter { it.isDirectory && !it.name.startsWith("tmp-") }
        .mapNotNull { runCatching { load(it.name) }.getOrNull() }

    /** The snapshot, after checking its files against the hashes and sizes it records and its id against its content. */
    fun load(snapshotId: String): Snapshot {
        if (!HEX64.matches(snapshotId)) throw DatasetImportException("DATASET_CORRUPT", "not a snapshot id")
        val dir = File(datasets, snapshotId)
        val meta = try {
            JsonParser.parseString(File(dir, "snapshot.json").readText()).asJsonObject
        } catch (e: Exception) {
            throw DatasetImportException("DATASET_CORRUPT", "snapshot.json is missing or unreadable")
        }
        val snapshot = try {
            Snapshot(
                snapshotId = meta.get("snapshotId").asString,
                dir = dir,
                modality = meta.get("modality").asString,
                inputShape = meta.getAsJsonArray("inputShape").map { it.asInt },
                inputDtype = meta.get("inputDtype").asString,
                classNames = meta.getAsJsonArray("classNames").map { it.asString },
                labelSchemaId = meta.get("labelSchemaId").asString,
                recordCount = meta.get("recordCount").asInt,
                inputsSha256 = meta.get("inputsSha256").asString,
                targetsSha256 = meta.get("targetsSha256").asString,
                transforms = meta.get("transforms")?.asString,
            )
        } catch (e: RuntimeException) {
            throw DatasetImportException("DATASET_CORRUPT", "snapshot.json is malformed")
        }
        val width = snapshot.inputShape.fold(1) { a, b -> a * b }
        val inputs = File(dir, "inputs.f32")
        val targets = File(dir, "targets.i64")
        val intact = snapshot.snapshotId == snapshotId &&
            Snapshot.idOf(snapshot.contentFields()) == snapshotId &&
            inputs.length() == snapshot.recordCount * width * 4L && targets.length() == snapshot.recordCount * 8L &&
            sha256Hex(inputs) == snapshot.inputsSha256 && sha256Hex(targets) == snapshot.targetsSha256
        if (!intact) throw DatasetImportException("DATASET_CORRUPT", "snapshot $snapshotId does not match its record")
        return snapshot
    }

    fun delete(snapshotId: String, pinned: Set<String>) {
        if (snapshotId in pinned) throw DatasetImportException("DATASET_PINNED", "snapshot $snapshotId is in use by a run")
        if (!HEX64.matches(snapshotId)) throw DatasetImportException("DATASET_CORRUPT", "not a snapshot id")
        File(datasets, snapshotId).deleteRecursively()
    }

    private companion object {
        val HEX64 = Regex("[0-9a-f]{64}")
    }
}
