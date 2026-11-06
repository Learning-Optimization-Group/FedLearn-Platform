package com.fedlearn.mobile.artifacts

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.network.OkHttpClientProvider
import java.util.concurrent.Executors

/**
 * JS entry point for Stage 3 artifact delivery (src/lib/artifactService.ts). Downloads run on a background thread
 * with credentials bound to the captured origin and no ambient cookie store. Files are written under the app's
 * private files directory. JS receives only the verified local path, or a rejection whose
 * code names the reason (ArtifactDownloadException.code).
 */
class ArtifactServiceModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {

  private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "artifact-download").apply { isDaemon = true } }

  override fun getName(): String = "ArtifactService"

  @ReactMethod
  fun fetchArtifact(url: String, sha256: String, byteSize: Double, credentialOrigin: String, token: String, promise: Promise) {
    executor.execute {
      try {
        val client = ArtifactHttpClient.create(OkHttpClientProvider.getOkHttpClient(), credentialOrigin, token)
        val downloader = ArtifactDownloader(client, ctx.filesDir)
        downloader.cleanQuarantine()
        val file = downloader.fetch(ArtifactDownloader.Descriptor(url, sha256, byteSize.toLong()))
        promise.resolve(file.absolutePath)
      } catch (e: ArtifactDownloadException) {
        promise.reject(e.code, e.code)
      } catch (e: Exception) {
        promise.reject("ARTIFACT_DOWNLOAD_FAILED", e.javaClass.simpleName)
      }
    }
  }
}
