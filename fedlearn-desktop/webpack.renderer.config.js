// Bypass Node 22+ SecurityError for html-webpack-plugin
Object.defineProperty(global, 'localStorage', {
  value: { getItem() {}, setItem() {}, removeItem() {} },
  writable: true,
});

const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const { buildRendererCsp } = require('./webpack.csp');

module.exports = {
  mode: 'development',
  target: 'web',
  entry: './src/renderer/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist/renderer'),
    filename: 'renderer.js',
    publicPath: '/',
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'],
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
      {
        test: /\.(png|svg|jpg|jpeg|gif|ico)$/i,
        type: 'asset/resource',
      },
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/i,
        type: 'asset/resource',
      },
    ],
  },
  plugins: [
    // Inject the real app version from package.json so the UI never ships a
    // stale hardcoded version string.
    new webpack.DefinePlugin({
      __APP_VERSION__: JSON.stringify(require('./package.json').version),
    }),
    new HtmlWebpackPlugin({
      template: './src/renderer/index.html',
      filename: 'index.html',
      // Dev build keeps 'unsafe-eval' — webpack's development build uses the
      // `eval` devtool, so the eval'd module wrappers need script-src to allow it.
      // It also keeps 'unsafe-inline' in style-src — style-loader injects
      // bundled CSS as literal <style> tags at runtime, required for HMR.
      templateParameters: {
        csp: buildRendererCsp({ allowEval: true, allowInlineStyle: true }),
      },
    }),
  ],
  devServer: {
    port: 9000,
    hot: true,
    static: {
      directory: path.resolve(__dirname, 'dist/renderer'),
    },
    headers: {
      // Applies when the dev server is hit directly in a browser tab (bypassing
      // Electron's own onHeadersReceived injection in src/main/main.ts). Fonts
      // are bundled locally (src/renderer/fonts.css) so no remote host is needed.
      'Content-Security-Policy':
        "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self' ws://localhost:9000 http://localhost:9000;",
    },
  },
};
