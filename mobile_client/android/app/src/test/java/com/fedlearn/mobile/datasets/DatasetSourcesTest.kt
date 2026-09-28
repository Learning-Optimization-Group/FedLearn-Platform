package com.fedlearn.mobile.datasets

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Stage 3 slice B2: the file a user picks is a CSV or a zipped dataset package. A package arrives as one archive
 * (the picker returns one file), so the archive is a boundary: only the two package entries, no path tricks, and a cap
 * on what it expands to.
 */
class DatasetSourcesTest {

    @get:Rule val tmp = TemporaryFolder()

    private val classes = listOf("c0", "c1", "c2")
    private val datasetJson = """{"schemaVersion":1,"modality":"vector","inputShape":[2],"classNames":["c0","c1","c2"],"recordCount":2}"""
    private val recordsJsonl = "{\"label\":\"c1\",\"values\":[1,2]}\n{\"label\":\"c2\",\"values\":[3,4]}\n"

    private fun sources(limits: DatasetImporter.Limits = DatasetImporter.Limits()) =
        DatasetSources(DatasetImporter(tmp.root, limits = limits), File(tmp.root, "cache"), limits)

    private fun zip(vararg entries: Pair<String, String>): ByteArray {
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { z ->
            for ((name, body) in entries) {
                z.putNextEntry(ZipEntry(name)); z.write(body.toByteArray()); z.closeEntry()
            }
        }
        return out.toByteArray()
    }

    private fun expect(code: String, block: () -> Unit) {
        try {
            block()
        } catch (e: DatasetImportException) {
            assertEquals(code, e.code)
            val leftovers = File(tmp.root, "cache").listFiles()?.toList() ?: emptyList()
            assertTrue("no extracted files are left behind: $leftovers", leftovers.isEmpty())
            return
        }
        fail("expected DatasetImportException $code")
    }

    @Test
    fun aCsvIsImportedAsACsv() {
        val s = sources().import("data.CSV", { ByteArrayInputStream("label,a,b\nc1,1,2\nc2,3,4\n".toByteArray()) }, classes, 2)
        assertEquals(2, s.recordCount)
    }

    @Test
    fun aZippedPackageIsTheSameSnapshotAsTheEquivalentCsv() {
        val fromZip = sources().import("mine.zip",
            { ByteArrayInputStream(zip("dataset.json" to datasetJson, "records.jsonl" to recordsJsonl)) }, classes, 2)
        val fromCsv = sources().import("mine.csv", { ByteArrayInputStream("label,a,b\nc1,1,2\nc2,3,4\n".toByteArray()) }, classes, 2)

        assertEquals(fromCsv.snapshotId, fromZip.snapshotId)
        assertTrue("extraction scratch is removed", File(tmp.root, "cache").listFiles().isNullOrEmpty())
    }

    @Test fun aPackageEntryInsideAFolderIsAccepted() {
        val s = sources().import("p.zip", { ByteArrayInputStream(zip("pkg/dataset.json" to datasetJson, "pkg/records.jsonl" to recordsJsonl)) }, classes, 2)
        assertEquals(2, s.recordCount)
    }

    @Test fun anArchiveWithAnExtraFileIsRefused() = expect("DATASET_BAD_PACKAGE") {
        sources().import("p.zip", { ByteArrayInputStream(zip("dataset.json" to datasetJson, "records.jsonl" to recordsJsonl, "run.sh" to "x")) }, classes, 2)
    }

    @Test fun anArchiveMissingItsRecordsIsRefused() = expect("DATASET_BAD_PACKAGE") {
        sources().import("p.zip", { ByteArrayInputStream(zip("dataset.json" to datasetJson)) }, classes, 2)
    }

    @Test fun anArchiveWithTwoDatasetFilesIsRefused() = expect("DATASET_BAD_PACKAGE") {
        sources().import("p.zip", { ByteArrayInputStream(zip("a/dataset.json" to datasetJson, "b/dataset.json" to datasetJson, "records.jsonl" to recordsJsonl)) }, classes, 2)
    }

    @Test fun anEntryThatClimbsOutOfTheArchiveIsRefused() = expect("DATASET_BAD_PACKAGE") {
        sources().import("p.zip", { ByteArrayInputStream(zip("../dataset.json" to datasetJson, "records.jsonl" to recordsJsonl)) }, classes, 2)
    }

    @Test fun anArchiveThatExpandsPastTheLimitIsRefused() = expect("DATASET_TOO_LARGE") {
        val big = "{\"label\":\"c0\",\"values\":[1,2]}\n".repeat(200)
        sources(DatasetImporter.Limits(maxSourceBytes = 1000)).import("p.zip",
            { ByteArrayInputStream(zip("dataset.json" to datasetJson, "records.jsonl" to big)) }, classes, 2)
    }

    // The importer caps each stream it reads, so only the archive's own cap bounds the total an archive expands to on
    // disk before the importer reads it: here each entry fits under the limit, and only together do they exceed it.
    @Test fun anArchiveWhoseEntriesTogetherExpandPastTheLimitIsRefused() = expect("DATASET_TOO_LARGE") {
        val paddedMeta = datasetJson + " ".repeat(600 - datasetJson.length)
        val records = "{\"label\":\"c0\",\"values\":[1,2]}\n".repeat(20)
        check(paddedMeta.length < 1000 && records.length < 1000 && paddedMeta.length + records.length > 1000)
        sources(DatasetImporter.Limits(maxSourceBytes = 1000)).import("p.zip",
            { ByteArrayInputStream(zip("dataset.json" to paddedMeta.replace("\"recordCount\":2", "\"recordCount\":20"), "records.jsonl" to records)) }, classes, 2)
    }

    @Test fun aNonArchiveNamedZipIsRefused() = expect("DATASET_BAD_PACKAGE") {
        sources().import("p.zip", { ByteArrayInputStream("not a zip".toByteArray()) }, classes, 2)
    }

    @Test fun anUnsupportedFileTypeIsRefused() = expect("DATASET_UNSUPPORTED_FORMAT") {
        sources().import("photo.jpg", { ByteArrayInputStream(ByteArray(3)) }, classes, 2)
    }
}
