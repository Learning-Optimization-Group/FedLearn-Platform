package com.fedlearn.mobile.artifacts

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.network.ForwardingCookieHandler
import com.facebook.react.modules.network.OkHttpClientProvider
import okhttp3.JavaNetCookieJar
import java.util.concurrent.Executors

/**
 * JS entry point for Stage 3 artifact delivery (src/lib/artifactService.ts). Downloads run on a background thread
 * through React Native's OkHttp client with the app's cookie store attached, so the session cookie the app's REST
 * calls hold applies. Files are written under the app's private files directory. JS receives only the verified local path, or a rejection whose
 * code names the reason (ArtifactDownloadException.code).
 */
class ArtifactServiceModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {

  private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "artifact-download").apply { isDaemon = true } }
  // The provider's shared client carries no cookie jar of its own: React Native's networking module attaches one
  // to the client it builds. Attach the same cookie store (ForwardingCookieHandler, backed by the app-wide
  // CookieManager where the REST session cookie lives), or every download is unauthenticated (403).
  private val downloader by lazy {
    val client = OkHttpClientProvider.getOkHttpClient().newBuilder()
      .cookieJar(JavaNetCookieJar(ForwardingCookieHandler(ctx)))
      .build()
    ArtifactDownloader(client, ctx.filesDir)
  }

  override fun getName(): String = "ArtifactService"

  @ReactMethod
  fun fetchArtifact(url: String, sha256: String, byteSize: Double, promise: Promise) {
    executor.execute {
      try {
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
