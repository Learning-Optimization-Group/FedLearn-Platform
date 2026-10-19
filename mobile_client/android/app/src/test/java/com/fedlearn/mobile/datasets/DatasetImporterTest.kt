package com.fedlearn.mobile.datasets

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/**
 * Stage 3 slice B1: a user's data becomes an immutable, content-addressed snapshot in the layout the native trainer
 * already reads (inputs.f32 row-major float32 + targets.i64), or is refused whole with a named reason.
 */
class DatasetImporterTest {

    @get:Rule val tmp = TemporaryFolder()

    private val tinyClasses = listOf("c0", "c1", "c2")

    private fun importer(space: Long = Long.MAX_VALUE, limits: DatasetImporter.Limits = DatasetImporter.Limits()) =
        DatasetImporter(tmp.root, usableSpace = { space }, limits = limits)

    private fun csv(text: String) = ByteArrayInputStream(text.toByteArray(Charsets.UTF_8))

    private fun expect(code: String, block: () -> Unit) {
        try {
            block()
        } catch (e: DatasetImportException) {
            assertEquals(code, e.code)
            assertNoPartialSnapshots()
            return
        }
        fail("expected DatasetImportException $code")
    }

    private fun assertNoPartialSnapshots() {
        val dir = File(tmp.root, "datasets")
        val leftovers = dir.listFiles()?.filter { it.name.startsWith("tmp-") } ?: emptyList()
        assertTrue("no partial import is left behind: $leftovers", leftovers.isEmpty())
    }

    private fun sha256(f: File) = MessageDigest.getInstance("SHA-256").digest(f.readBytes()).joinToString("") { "%02x".format(it) }

    // --- label schema: byte-identical to the contract's (Python execution_plan.label_schema_id) ---------------

    @Test
    fun theLabelSchemaIdIsTheContractsDefinition() {
        assertEquals("labels-sha256:4817507d9942bf24635535b19b1588bb44e00d2b51314536926f3580c7f5d896",
            LabelSchema.id(tinyClasses))
        assertEquals("labels-sha256:088438465b960fb54890fd9d6001368f3be685bf335dd0b14db3428401810779",
            LabelSchema.id(listOf("café", "naïve \"q\"", "tab\there", "日本")))
    }

    // --- CSV ------------------------------------------------------------------------------------------------

    @Test
    fun aCsvBecomesASnapshotInTheNativeLayout() {
        val s = importer().importCsv(csv("label,a,b,c,d\nc1,1,2,3,4\nc0,-0.5,0,1e-3,7\n"), tinyClasses, inputWidth = 4)

        assertEquals(2, s.recordCount)
        assertEquals(listOf(4), s.inputShape)
        assertEquals(LabelSchema.id(tinyClasses), s.labelSchemaId)
        assertEquals(File(tmp.root, "datasets/${s.snapshotId}"), s.dir)
        val floats = ByteBuffer.wrap(File(s.dir, "inputs.f32").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val got = FloatArray(floats.remaining()).also { floats.get(it) }
        assertArrayEquals(floatArrayOf(1f, 2f, 3f, 4f, -0.5f, 0f, 1e-3f, 7f), got, 0f)
        val longs = ByteBuffer.wrap(File(s.dir, "targets.i64").readBytes()).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer()
        assertEquals(listOf(1L, 0L), List(longs.remaining()) { longs.get() })
        assertEquals(sha256(File(s.dir, "inputs.f32")), s.inputsSha256)
        assertEquals(sha256(File(s.dir, "targets.i64")), s.targetsSha256)
    }

    @Test
    fun importingTheSameContentTwiceIsTheSameSnapshot() {
        val a = importer().importCsv(csv("label,x\nc2,1\n"), tinyClasses, 1)
        val b = importer().importCsv(csv("label,x\nc2,1\n"), tinyClasses, 1)

        assertEquals(a.snapshotId, b.snapshotId)
        assertEquals(1, File(tmp.root, "datasets").listFiles()!!.count { !it.name.startsWith("tmp-") })
    }

    @Test fun aHeaderWithoutALabelColumnIsRefused() =
        expect("DATASET_BAD_HEADER") { importer().importCsv(csv("x,y\n1,2\n"), tinyClasses, 1) }

    @Test fun aHeaderWithTheWrongFeatureCountIsRefused() =
        expect("DATASET_SCHEMA_MISMATCH") { importer().importCsv(csv("label,x,y\nc0,1,2\n"), tinyClasses, 3) }

    @Test fun aRowWithTheWrongColumnCountIsRefused() =
        expect("DATASET_BAD_ROW") { importer().importCsv(csv("label,x,y\nc0,1\n"), tinyClasses, 2) }

    @Test fun aQuotedFieldIsRefused() =
        expect("DATASET_BAD_ROW") { importer().importCsv(csv("label,x\n\"c0\",1\n"), tinyClasses, 1) }

    @Test fun aNonNumericValueIsRefused() =
        expect("DATASET_BAD_VALUE") { importer().importCsv(csv("label,x\nc0,0x1p3\n"), tinyClasses, 1) }

    @Test fun aNanIsRefused() =
        expect("DATASET_NON_FINITE") { importer().importCsv(csv("label,x\nc0,NaN\n"), tinyClasses, 1) }

    @Test fun anOverflowingValueIsRefused() =
        expect("DATASET_NON_FINITE") { importer().importCsv(csv("label,x\nc0,1e999\n"), tinyClasses, 1) }

    @Test fun aLabelOutsideTheClassListIsRefused() =
        expect("DATASET_UNKNOWN_LABEL") { importer().importCsv(csv("label,x\nc9,1\n"), tinyClasses, 1) }

    @Test fun aHeaderOnlyFileIsRefused() =
        expect("DATASET_EMPTY") { importer().importCsv(csv("label,x\n"), tinyClasses, 1) }

    @Test fun tooManyRecordsAreRefused() = expect("DATASET_TOO_MANY_RECORDS") {
        importer(limits = DatasetImporter.Limits(maxRecords = 2)).importCsv(csv("label,x\nc0,1\nc0,1\nc0,1\n"), tinyClasses, 1)
    }

    @Test fun anOversizedSourceIsRefused() = expect("DATASET_TOO_LARGE") {
        importer(limits = DatasetImporter.Limits(maxSourceBytes = 20)).importCsv(csv("label,x\nc0,1\nc0,1\nc0,1\n"), tinyClasses, 1)
    }

    @Test fun invalidUtf8IsRefused() = expect("DATASET_NOT_UTF8") {
        val bytes = "label,x\nc0,1\n".toByteArray() + byteArrayOf(0xC3.toByte(), 0x28)
        importer().importCsv(ByteArrayInputStream(bytes), tinyClasses, 1)
    }

    @Test fun insufficientStorageIsRefused() =
        expect("INSUFFICIENT_STORAGE") { importer(space = 1000).importCsv(csv("label,x\nc0,1\n"), tinyClasses, 1) }

    // --- package (dataset.json + records.jsonl) ---------------------------------------------------------------

    private fun pkg(classes: List<String> = tinyClasses, modality: String = "vector", count: Int = 2, width: Int = 2) =
        ByteArrayInputStream("""{"schemaVersion":1,"modality":"$modality","inputShape":[$width],
            "classNames":${classes.joinToString(",", "[", "]") { "\"$it\"" }},"recordCount":$count}""".toByteArray())

    private fun records(text: String) = ByteArrayInputStream(text.toByteArray())

    @Test
    fun aPackageAndTheEquivalentCsvAreTheSameSnapshot() {
        val fromCsv = importer().importCsv(csv("label,a,b\nc1,1,2\nc2,3,4\n"), tinyClasses, 2)
        val fromPkg = importer().importPackage(pkg(),
            records("{\"label\":\"c1\",\"values\":[1,2]}\n{\"label\":\"c2\",\"values\":[3,4]}\n"), tinyClasses, 2)

        assertEquals(fromCsv.snapshotId, fromPkg.snapshotId)
    }

    @Test fun aPackageForOtherClassesIsRefused() = expect("DATASET_SCHEMA_MISMATCH") {
        importer().importPackage(pkg(classes = listOf("x", "y", "z")), records("{\"label\":\"x\",\"values\":[1,2]}\n"), tinyClasses, 2)
    }

    @Test fun aNonVectorPackageIsRefusedInStage3() = expect("DATASET_BAD_PACKAGE") {
        importer().importPackage(pkg(modality = "image"), records(""), tinyClasses, 2)
    }

    @Test fun aPackageWhoseRecordCountIsWrongIsRefused() = expect("DATASET_BAD_PACKAGE") {
        importer().importPackage(pkg(count = 3), records("{\"label\":\"c0\",\"values\":[1,2]}\n"), tinyClasses, 2)
    }

    @Test fun aMalformedRecordLineIsRefused() = expect("DATASET_BAD_ROW") {
        importer().importPackage(pkg(count = 1), records("{\"label\":\"c0\",\"values\":[1,2]\n"), tinyClasses, 2)
    }

    @Test fun aPackageRecordWithANanIsRefused() = expect("DATASET_NON_FINITE") {
        importer().importPackage(pkg(count = 1), records("{\"label\":\"c0\",\"values\":[1,\"NaN\"]}\n"), tinyClasses, 2)
    }

    // --- store: load verifies, delete respects pins -----------------------------------------------------------

    @Test
    fun aLoadedSnapshotIsVerifiedAgainstItsRecordedHashes() {
        val s = importer().importCsv(csv("label,x\nc0,1\n"), tinyClasses, 1)
        val store = DatasetStore(tmp.root)
        assertEquals(s, store.load(s.snapshotId))

        File(s.dir, "inputs.f32").writeBytes(ByteArray(4))
        try {
            store.load(s.snapshotId)
            fail("expected DATASET_CORRUPT")
        } catch (e: DatasetImportException) {
            assertEquals("DATASET_CORRUPT", e.code)
        }
    }

    @Test
    fun aPinnedSnapshotCannotBeDeletedAndAnUnpinnedOneCan() {
        val s = importer().importCsv(csv("label,x\nc0,1\n"), tinyClasses, 1)
        val store = DatasetStore(tmp.root)
        try {
            store.delete(s.snapshotId, pinned = setOf(s.snapshotId))
            fail("expected DATASET_PINNED")
        } catch (e: DatasetImportException) {
            assertEquals("DATASET_PINNED", e.code)
        }
        assertTrue(s.dir.exists())

        store.delete(s.snapshotId, pinned = emptySet())

        assertFalse(s.dir.exists())
        assertEquals(emptyList<Snapshot>(), store.list())
    }
}
