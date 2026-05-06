// Bypass Node 22+ SecurityError for html-webpack-plugin
Object.defineProperty(global, 'localStorage', {
  value: { getItem() {}, setItem() {}, removeItem() {} },
  writable: true,
});

const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');

// =============================================================================
// FedLearn Desktop — Production Webpack Configuration
// =============================================================================
// Builds all three Electron targets (main, preload, renderer) in production mode.
// TerserPlugin with drop_console is applied to the Renderer and Preload bundles
// to strip all console.* calls from the production binary, preventing JWT/path
// leakage to Chromium DevTools (Section 5.3 of the deployment guide).
// =============================================================================

const terserWithConsoleStrip = new TerserPlugin({
  terserOptions: {
    compress: {
      drop_console: true,
      drop_debugger: true,
      pure_funcs: ['console.log', 'console.info', 'console.debug', 'console.warn'],
    },
    output: {
      comments: false,
    },
  },
  extractComments: false,
});

const terserMainProcess = new TerserPlugin({
  terserOptions: {
    compress: {
      drop_debugger: true,
      // Main process retains electron-log calls; console.* is NOT stripped
      // because electron-log may internally use console as a transport fallback.
    },
    output: {
      comments: false,
    },
  },
  extractComments: false,
});

// --- Main Process ---
const mainConfig = {
  name: 'main',
  mode: 'production',
  target: 'electron-main',
  entry: './src/main/main.ts',
  output: {
    path: path.resolve(__dirname, 'dist/main'),
    filename: 'main.js',
  },
  resolve: {
    extensions: ['.ts', '.js', '.json'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  externals: {
    dockerode: 'commonjs dockerode',
    'electron-store': 'commonjs electron-store',
    'electron-log': 'commonjs electron-log',
    electron: 'commonjs electron',
  },
  node: {
    __dirname: false,
    __filename: false,
  },
  optimization: {
    minimize: true,
    minimizer: [terserMainProcess],
  },
  devtool: false,
};

// --- Preload Script ---
const preloadConfig = {
  name: 'preload',
  mode: 'production',
  target: 'electron-preload',
  entry: './src/preload/preload.ts',
  output: {
    path: path.resolve(__dirname, 'dist/preload'),
    filename: 'preload.js',
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  externals: {
    electron: 'commonjs electron',
  },
  node: {
    __dirname: false,
    __filename: false,
  },
  optimization: {
    minimize: true,
    minimizer: [terserWithConsoleStrip],
  },
  devtool: false,
};

// --- Renderer Process ---
const rendererConfig = {
  name: 'renderer',
  mode: 'production',
  target: 'web',
  entry: './src/renderer/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist/renderer'),
    filename: 'renderer.[contenthash].js',
    publicPath: './',
    clean: true,
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
    new HtmlWebpackPlugin({
      template: './src/renderer/index.html',
      filename: 'index.html',
      minify: {
        collapseWhitespace: true,
        removeComments: true,
        removeRedundantAttributes: true,
      },
    }),
  ],
  optimization: {
    minimize: true,
    minimizer: [terserWithConsoleStrip],
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendor',
          chunks: 'all',
        },
      },
    },
  },
  devtool: false,
};

module.exports = [mainConfig, preloadConfig, rendererConfig];
