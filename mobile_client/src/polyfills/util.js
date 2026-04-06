/**
 * Custom util polyfill for React Native
 * 
 * This module provides a complete util polyfill that includes isTypedArray,
 * which is required by TensorFlow.js but missing from the npm util package.
 * 
 * Metro's extraNodeModules will resolve ALL require('util') calls to this module,
 * ensuring TensorFlow.js gets isTypedArray when it needs it.
 */

// Import the npm util package to get all standard util functions
import * as npmUtil from 'util';

// Create a complete util object with all npm util functions
const util = { ...npmUtil };

// Add isTypedArray function (Node.js built-in that npm package lacks)
// This is critical for TensorFlow.js which uses util.isTypedArray internally
util.isTypedArray = function (value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  return (
    value instanceof Int8Array ||
    value instanceof Uint8Array ||
    value instanceof Uint8ClampedArray ||
    value instanceof Int16Array ||
    value instanceof Uint16Array ||
    value instanceof Int32Array ||
    value instanceof Uint32Array ||
    value instanceof Float32Array ||
    value instanceof Float64Array ||
    (typeof BigInt64Array !== 'undefined' &&
      value instanceof BigInt64Array) ||
    (typeof BigUint64Array !== 'undefined' && value instanceof BigUint64Array)
  );
};

// Export the complete util object
// When TensorFlow.js does require('util'), it will get this module
export default util;

// Also export as named exports for compatibility
export const {
  inspect,
  format,
  formatWithOptions,
  log,
  deprecate,
  debuglog,
  isArray,
  isRegExp,
  isDate,
  isError,
  inherits,
  _extend,
  promisify,
  callbackify,
  types,
  TextEncoder,
  TextDecoder,
  isTypedArray, // Export isTypedArray as a named export
} = util;

