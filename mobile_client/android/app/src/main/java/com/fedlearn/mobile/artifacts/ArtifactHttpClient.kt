package com.fedlearn.mobile.artifacts

import okhttp3.CookieJar
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient

/** Request-specific authority; downloads never inherit an ambient browser cookie or follow a redirect. */
object ArtifactHttpClient {
    fun create(base: OkHttpClient, credentialOrigin: String, token: String): OkHttpClient {
        val scope = credentialOrigin.toHttpUrlOrNull()
            ?: throw ArtifactDownloadException("ARTIFACT_AUTH_SCOPE_MISMATCH", "Invalid credential origin")
        if (scope.username.isNotEmpty() || scope.password.isNotEmpty() || scope.encodedPath != "/"
            || scope.query != null || scope.fragment != null || token.any { it <= ' ' || it == '\u007f' }) {
            throw ArtifactDownloadException("ARTIFACT_AUTH_SCOPE_MISMATCH", "Invalid credential scope")
        }
        return base.newBuilder()
            .cookieJar(CookieJar.NO_COOKIES)
            .followRedirects(false)
            .followSslRedirects(false)
            .addInterceptor { chain ->
                val request = chain.request()
                val target = request.url
                if (target.username.isNotEmpty() || target.password.isNotEmpty()
                    || (token.isNotEmpty() && (target.scheme != scope.scheme || target.host != scope.host || target.port != scope.port))) {
                    throw ArtifactDownloadException("ARTIFACT_AUTH_SCOPE_MISMATCH", "Credential does not authorize this origin")
                }
                val builder = request.newBuilder().removeHeader("Cookie").removeHeader("Authorization")
                    .header("X-FedLearn-Client", "fedlearn-mobile")
                if (token.isNotEmpty()) builder.header("Authorization", "Bearer $token")
                chain.proceed(builder.build())
            }
            .build()
    }
}
