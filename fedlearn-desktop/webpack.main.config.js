const path = require('path');

module.exports = {
  mode: 'development',
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
    // dockerode uses native bindings; keep as external require
    dockerode: 'commonjs dockerode',
    'electron-store': 'commonjs electron-store',
    'electron-log': 'commonjs electron-log',
    electron: 'commonjs electron',
  },
  node: {
    __dirname: false,
    __filename: false,
  },
};
