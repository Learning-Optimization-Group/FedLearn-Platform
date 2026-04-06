// polyfills.js - Minimal polyfills for React Native

if (!global.process) {
  global.process = require('process/browser');
}

if (!global.process.env) {
  global.process.env = {};
}
