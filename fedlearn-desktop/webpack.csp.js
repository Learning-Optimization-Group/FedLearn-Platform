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
//
// style-src follows the same dev/prod split: dev uses style-loader (runtime
// <style> tag injection, required for HMR) so it needs 'unsafe-inline'; prod
// extracts CSS via MiniCssExtractPlugin into a real .css file loaded through
// a <link rel="stylesheet">, so no inline styles are ever injected and
// 'unsafe-inline' can be dropped from style-src.
// =============================================================================

function buildRendererCsp({ allowEval, allowInlineStyle }) {
  const scriptSrc = allowEval ? "script-src 'self' 'unsafe-eval'" : "script-src 'self'";
  const styleSrc = allowInlineStyle ? "style-src 'self' 'unsafe-inline'" : "style-src 'self'";
  return [
    "default-src 'self'",
    scriptSrc,
    styleSrc,
    "font-src 'self'",
    "img-src 'self' data:",
    "connect-src 'self' http://localhost:* https://localhost:* ws://localhost:* wss://localhost:*",
    "frame-src 'none'",
    "object-src 'none'",
    "base-uri 'self'",
  ].join('; ');
}

module.exports = { buildRendererCsp };
