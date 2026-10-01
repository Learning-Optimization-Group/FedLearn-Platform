package com.fedlearn.mobile.datasets

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

/**
 * Stage 4 S5: a user's images arrive as raw 8-bit pixels in a dataset package and become a float32 CHW snapshot through
 * the contract's ImageToUnitTensor and NormalizeChannels. The tensors must be the Python reference's to the bit
 * (image_transforms_v1.golden, itself checked against torchvision), or a phone's update would stop being replayable.
 */
class ImageImportTest {

    @get:Rule val tmp = TemporaryFolder()

    private val classes = listOf("cat", "dog")
    private val half3 = listOf(0.5f, 0.5f, 0.5f)

    private fun importer() = DatasetImporter(tmp.root, usableSpace = { Long.MAX_VALUE })

    private fun stream(text: String) = ByteArrayInputStream(text.toByteArray(Charsets.UTF_8))

    private fun meta(h: Int, w: Int, c: Int, count: Int, modality: String = "image") =
        """{"schemaVersion":1,"modality":"$modality","height":$h,"width":$w,"channels":$c,""" +
            """"classNames":["cat","dog"],"recordCount":$count}"""

    private fun record(label: String, pixels: ByteArray) =
        """{"label":"$label","pixels":"${Base64.getEncoder().encodeToString(pixels)}"}"""

    private fun pixels(n: Int, seed: Int) = ByteArray(n) { ((it * 37 + seed * 101) % 256).toByte() }

    private fun expect(code: String, block: () -> Unit) {
        try {
            block()
        } catch (e: DatasetImportException) {
            assertEquals(code, e.code)
            return
        }
        fail("expected DatasetImportException $code")
    }

    private fun floats(file: File): FloatArray {
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(buf.remaining() / 4) { buf.getFloat() }
    }

    // --- the arithmetic: the golden, bit for bit -------------------------------------------------------------

    @Test
    fun theTransformsReproduceTheReferenceGoldenBitForBit() {
        val golden = File("../../../framework/tests/fixtures/execution_contract_v1/image_transforms_v1.golden")
        val cases = golden.readLines().filter { it.startsWith("case ") }
        assertEquals(6, cases.size)
        for (line in cases) {
            val (head, pixelHex, wordsHex) = line.split(" : ")
            val parts = head.split(' ')
            val (h, w, c) = parts.subList(2, 5).map { it.toInt() }
            val mean = parts[5].removePrefix("mean=").split(',').filter { it.isNotEmpty() }.map { it.toFloat() }
            val std = parts[6].removePrefix("std=").split(',').filter { it.isNotEmpty() }.map { it.toFloat() }
            val px = pixelHex.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
            val shape = DataShape.Image(h, w, c, mean.takeIf { it.isNotEmpty() }, std.takeIf { it.isNotEmpty() })
            val got = ImageTransforms.toTensor(px, shape).map { java.lang.Float.floatToRawIntBits(it) }
            val want = wordsHex.split(' ').map { java.lang.Long.parseLong(it, 16).toInt() }
            assertEquals("case ${parts[1]}", want, got)
        }
    }

    @Test
    fun aPackageWrittenByThePythonToolBecomesExactlyTheReferenceTensors() {
        // framework/tests/fixtures/image_package_v1: written by fedlearn.contract.image_package, with the tensors the
        // Python reference makes of it. The phone must write the same bytes, so the format is pinned on both sides.
        val dir = File("../../../framework/tests/fixtures/image_package_v1")
        val snapshot = DatasetSources(importer(), tmp.newFolder()).import("tiny.zip",
            { File(dir, "tiny.zip").inputStream() }, listOf("cat", "dog"),
            DataShape.Image(4, 5, 3, half3, half3))
        assertEquals(3, snapshot.recordCount)
        assertEquals(java.security.MessageDigest.getInstance("SHA-256").digest(File(dir, "tiny_inputs.f32").readBytes())
            .joinToString("") { "%02x".format(it) }, snapshot.inputsSha256)
        assertTrue(File(snapshot.dir, "inputs.f32").readBytes().contentEquals(File(dir, "tiny_inputs.f32").readBytes()))
        assertTrue(File(snapshot.dir, "targets.i64").readBytes().contentEquals(File(dir, "tiny_targets.i64").readBytes()))
    }

    // --- the package ------------------------------------------------------------------------------------------

    @Test
    fun anImagePackageBecomesAChwFloatSnapshot() {
        val shape = DataShape.Image(2, 3, 3, half3, half3)
        val a = pixels(18, 1)
        val b = pixels(18, 2)
        val snapshot = importer().importImagePackage(stream(meta(2, 3, 3, 2)),
            stream(record("dog", a) + "\n" + record("cat", b) + "\n"), classes, shape)
        assertEquals("image", snapshot.modality)
        assertEquals(listOf(3, 2, 3), snapshot.inputShape)
        assertEquals(2, snapshot.recordCount)
        assertEquals(shape.transformsId, snapshot.transforms)
        val expected = ImageTransforms.toTensor(a, shape) + ImageTransforms.toTensor(b, shape)
        assertEquals(expected.toList(), floats(File(snapshot.dir, "inputs.f32")).toList())
        val targets = ByteBuffer.wrap(File(snapshot.dir, "targets.i64").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        assertEquals(listOf(1L, 0L), listOf(targets.getLong(), targets.getLong()))
    }

    @Test
    fun theSnapshotRecordsHowItsValuesWerePrepared() {
        val px = pixels(18, 3)
        fun importWith(shape: DataShape.Image) = DatasetImporter(tmp.newFolder(), usableSpace = { Long.MAX_VALUE })
            .importImagePackage(stream(meta(2, 3, 3, 1)), stream(record("cat", px)), classes, shape)
        val normalised = importWith(DataShape.Image(2, 3, 3, half3, half3))
        val raw = importWith(DataShape.Image(2, 3, 3, null, null))
        assertNotEquals(normalised.transforms, raw.transforms)
        assertNotEquals(normalised.snapshotId, raw.snapshotId)
        assertEquals("image:2x3x3;mean=3f000000,3f000000,3f000000;std=3f000000,3f000000,3f000000",
            normalised.transforms)
        assertEquals("image:2x3x3", raw.transforms)
    }

    @Test
    fun aVectorSnapshotRecordsNoPreparationAndKeepsItsId() {
        val snapshot = importer().importCsv(stream("f0,f1,label\n1,2,cat\n"), classes, 2)
        assertEquals(null, snapshot.transforms)
        assertTrue(!snapshot.contentFields().containsKey("transforms"))
    }

    @Test
    fun anImageOfTheWrongSizeIsRefused() {
        expect("DATASET_BAD_ROW") {
            importer().importImagePackage(stream(meta(2, 3, 3, 1)), stream(record("cat", pixels(17, 1))), classes,
                DataShape.Image(2, 3, 3, null, null))
        }
    }

    @Test
    fun pixelsThatAreNotBase64AreRefused() {
        expect("DATASET_BAD_ROW") {
            importer().importImagePackage(stream(meta(2, 3, 3, 1)), stream("""{"label":"cat","pixels":"%%%"}"""),
                classes, DataShape.Image(2, 3, 3, null, null))
        }
    }

    @Test
    fun aPackageOfAnotherImageSizeIsRefused() {
        expect("DATASET_SCHEMA_MISMATCH") {
            importer().importImagePackage(stream(meta(3, 2, 3, 1)), stream(record("cat", pixels(18, 1))), classes,
                DataShape.Image(2, 3, 3, null, null))
        }
    }

    @Test
    fun aVectorPackageIsNotAnImagePackage() {
        expect("DATASET_SCHEMA_MISMATCH") {
            importer().importImagePackage(stream(meta(2, 3, 3, 1, modality = "vector")),
                stream(record("cat", pixels(18, 1))), classes, DataShape.Image(2, 3, 3, null, null))
        }
    }

    @Test
    fun anImagePackageIsRefusedByAVectorRun() {
        expect("DATASET_BAD_PACKAGE") {
            importer().importPackage(stream(meta(2, 3, 3, 1)), stream(record("cat", pixels(18, 1))), classes, 18)
        }
    }

    @Test
    fun aCsvCannotCarryImages() {
        val sources = DatasetSources(importer(), tmp.newFolder())
        expect("DATASET_UNSUPPORTED_FORMAT") {
            sources.import("pets.csv", { stream("f0,label\n1,cat\n") }, classes, DataShape.Image(2, 3, 3, null, null))
        }
    }
}
