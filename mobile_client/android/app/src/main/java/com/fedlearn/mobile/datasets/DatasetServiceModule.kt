package com.fedlearn.mobile.datasets

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.BaseActivityEventListener
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.WritableMap
import java.io.File
import java.util.concurrent.Executors

/**
 * JS entry point for on-device datasets (src/lib/datasetService.ts, Stage 3 slice B2). The user picks one file with
 * the system picker (Storage Access Framework, read-only, no persistent grant); it is imported on a background thread
 * into an immutable snapshot under the app's private files directory. Nothing about the data leaves the device: JS
 * receives the snapshot's metadata and the local paths the native trainer reads.
 */
class DatasetServiceModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {

  private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "dataset-import").apply { isDaemon = true } }
  private val store by lazy { DatasetStore(ctx.filesDir) }
  private val sources by lazy { DatasetSources(DatasetImporter(ctx.filesDir), File(ctx.cacheDir, "dataset-import")) }

  private var pending: PendingPick? = null

  private class PendingPick(val promise: Promise, val classNames: List<String>, val inputWidth: Int)

  private val results = object : BaseActivityEventListener() {
    override fun onActivityResult(activity: Activity, requestCode: Int, resultCode: Int, data: Intent?) {
      if (requestCode != PICK_REQUEST) return
      val pick = pending ?: return
      pending = null
      val uri = data?.data
      if (resultCode != Activity.RESULT_OK || uri == null) {
        pick.promise.reject("DATASET_PICK_CANCELLED", "no file was chosen")
        return
      }
      executor.execute { importUri(uri, pick) }
    }
  }

  init {
    ctx.addActivityEventListener(results)
  }

  override fun getName(): String = "DatasetService"

  /** Open the system picker for a .csv or a zipped package, and import the chosen file against the run's classes. */
  @ReactMethod
  fun pickAndImport(classNames: ReadableArray, inputWidth: Double, promise: Promise) {
    val activity = currentActivity
    if (activity == null) {
      promise.reject("DATASET_PICK_UNAVAILABLE", "the app is not in the foreground")
      return
    }
    if (pending != null) {
      promise.reject("DATASET_PICK_BUSY", "a file is already being chosen")
      return
    }
    pending = PendingPick(promise, (0 until classNames.size()).map { classNames.getString(it) ?: "" }, inputWidth.toInt())
    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = "*/*"
      putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("text/csv", "text/comma-separated-values", "text/plain",
        "application/zip", "application/x-zip-compressed", "application/octet-stream"))
    }
    activity.startActivityForResult(intent, PICK_REQUEST)
  }

  @ReactMethod
  fun list(promise: Promise) {
    executor.execute {
      val array = Arguments.createArray()
      store.list().forEach { array.pushMap(toMap(it)) }
      promise.resolve(array)
    }
  }

  @ReactMethod
  fun delete(snapshotId: String, pinned: ReadableArray, promise: Promise) {
    executor.execute {
      try {
        store.delete(snapshotId, (0 until pinned.size()).mapNotNull { pinned.getString(it) }.toSet())
        promise.resolve(null)
      } catch (e: DatasetImportException) {
        promise.reject(e.code, e.message)
      }
    }
  }

  private fun importUri(uri: Uri, pick: PendingPick) {
    try {
      val resolver = ctx.contentResolver
      val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) c.getString(0) else null
      } ?: uri.lastPathSegment ?: ""
      val snapshot = sources.import(name, {
        resolver.openInputStream(uri) ?: throw DatasetImportException("DATASET_UNREADABLE", "the file could not be opened")
      }, pick.classNames, pick.inputWidth)
      pick.promise.resolve(toMap(snapshot))
    } catch (e: DatasetImportException) {
      pick.promise.reject(e.code, e.message)
    } catch (e: Exception) {
      pick.promise.reject("DATASET_IMPORT_FAILED", e.javaClass.simpleName)
    }
  }

  private fun toMap(s: Snapshot): WritableMap = Arguments.createMap().apply {
    putString("snapshotId", s.snapshotId)
    putInt("recordCount", s.recordCount)
    putArray("inputShape", Arguments.createArray().apply { s.inputShape.forEach { pushInt(it) } })
    putString("inputDtype", s.inputDtype)
    putArray("classNames", Arguments.createArray().apply { s.classNames.forEach { pushString(it) } })
    putString("labelSchemaId", s.labelSchemaId)
    putString("inputsPath", File(s.dir, "inputs.f32").absolutePath)
    putString("targetsPath", File(s.dir, "targets.i64").absolutePath)
  }

  private companion object {
    const val PICK_REQUEST = 0x5d47
  }
}
