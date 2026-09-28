package com.fedlearn.mobile.datasets

import java.io.File
import java.io.InputStream
import java.util.UUID
import java.util.zip.ZipException
import java.util.zip.ZipInputStream

/**
 * Imports the file a user picked (Stage 3 slice B2): a `.csv`, or a dataset package zipped into one `.zip`, because
 * the system picker returns a single file. The archive is a trust boundary. It may hold exactly one `dataset.json` and
 * one `records.jsonl` (at any folder depth), plus nothing else except the `__MACOSX/` entries macOS adds. No entry may
 * climb out of the archive, and what it expands to is capped by the importer's source limit. Entries are extracted to a
 * scratch directory that is always removed.
 */
class DatasetSources(
    private val importer: DatasetImporter,
    private val scratchRoot: File,
    private val limits: DatasetImporter.Limits = DatasetImporter.Limits(),
) {
    fun import(displayName: String, open: () -> InputStream, classNames: List<String>, inputWidth: Int): Snapshot {
        val name = displayName.lowercase()
        return when {
            name.endsWith(".csv") -> open().use { importer.importCsv(it, classNames, inputWidth) }
            name.endsWith(".zip") -> importZip(open, classNames, inputWidth)
            else -> throw DatasetImportException("DATASET_UNSUPPORTED_FORMAT", "choose a .csv file or a zipped dataset package")
        }
    }

    private fun importZip(open: () -> InputStream, classNames: List<String>, inputWidth: Int): Snapshot {
        scratchRoot.mkdirs()
        val dir = File(scratchRoot, "unzip-${UUID.randomUUID()}").apply { mkdirs() }
        try {
            var datasetJson: File? = null
            var recordsJsonl: File? = null
            var expanded = 0L
            var entries = 0
            try {
                ZipInputStream(open()).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        if (++entries > MAX_ENTRIES) bad("the archive has too many entries")
                        val path = entry.name
                        if (path.startsWith("__MACOSX/")) continue
                        if (path.startsWith("/") || path.contains('\\') || path.split('/').any { it == ".." }) {
                            bad("an archive entry points outside the archive")
                        }
                        if (entry.isDirectory) continue
                        val target = when (path.substringAfterLast('/')) {
                            "dataset.json" -> if (datasetJson == null) File(dir, "dataset.json").also { datasetJson = it } else bad("two dataset.json files")
                            "records.jsonl" -> if (recordsJsonl == null) File(dir, "records.jsonl").also { recordsJsonl = it } else bad("two records.jsonl files")
                            else -> bad("unexpected file in the package")
                        }
                        target.outputStream().use { out ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                val n = zip.read(buffer)
                                if (n < 0) break
                                expanded += n
                                if (expanded > limits.maxSourceBytes) {
                                    throw DatasetImportException("DATASET_TOO_LARGE", "the package expands past ${limits.maxSourceBytes} bytes")
                                }
                                out.write(buffer, 0, n)
                            }
                        }
                    }
                }
            } catch (e: ZipException) {
                bad("not a readable zip archive")
            }
            val meta = datasetJson ?: bad("the package has no dataset.json")
            val records = recordsJsonl ?: bad("the package has no records.jsonl")
            return meta.inputStream().use { m ->
                records.inputStream().use { r -> importer.importPackage(m, r, classNames, inputWidth) }
            }
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun bad(reason: String): Nothing = throw DatasetImportException("DATASET_BAD_PACKAGE", reason)

    private companion object {
        const val MAX_ENTRIES = 64
    }
}
