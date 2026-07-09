// =============================================================================
// FedLearn Desktop — Renderer Content-Security-Policy builder (DE-14)
// =============================================================================
// Single source of truth for the <meta> CSP baked into src/renderer/index.html
// by HtmlWebpackPlugin (via templateParameters, see webpack.renderer.config.js
// and webpack.prod.config.js). Dev needs 'unsafe-eval' because webpack's
// development build uses the `eval` devtool; the packaged production build
// must not carry it. Both configs otherwise render the exact same policy —
// no remote font hosts, since Bricolage Grotesque, Hanken Grotesk, and
// JetBrains Mono are bundled locally (see src/renderer/fonts.css) instead of
// loaded from Google Fonts.
// =============================================================================

function buildRendererCsp({ allowEval }) {
  const scriptSrc = allowEval ? "script-src 'self' 'unsafe-eval'" : "script-src 'self'";
  return [
    "default-src 'self'",
    scriptSrc,
    // 'unsafe-inline' is required regardless of environment: style-loader
    // injects bundled CSS (styles.css, tokens.css, the local font faces) as
    // literal <style> tags at runtime.
    "style-src 'self' 'unsafe-inline'",
    "font-src 'self'",
    "img-src 'self' data:",
    "connect-src 'self' http://localhost:* https://localhost:* ws://localhost:* wss://localhost:*",
    "frame-src 'none'",
    "object-src 'none'",
    "base-uri 'self'",
  ].join('; ');
}

module.exports = { buildRendererCsp };
