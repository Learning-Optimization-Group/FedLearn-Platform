/**
 * Resource Monitor for FedMob Mobile Client
 * Tracks memory usage, training performance, and device state
 */

import * as tf from '@tensorflow/tfjs';

class ResourceMonitor {
  constructor() {
    this.metrics = {
      memory: {},
      training: {},
      device: {},
    };

    this.startTime = Date.now();
    this.lastUpdate = this.startTime;
  }

  /**
   * Get TensorFlow.js memory info
   * @returns {Object} Memory statistics
   */
  async getTFMemoryInfo() {
    const memoryInfo = await tf.memory();
    return {
      numTensors: memoryInfo.numTensors,
      numDataBuffers: memoryInfo.numDataBuffers,
      bytesInGPU: memoryInfo.numBytesInGPU,
      bytesInMemory: memoryInfo.numBytes,
      unreliable: memoryInfo.unreliable,
      reasons: memoryInfo.reasons,
    };
  }

  /**
   * Update training metrics
   * @param {Object} metrics - Training metrics
   */
  updateTrainingMetrics(metrics) {
    const now = Date.now();
    this.metrics.training = {
      ...metrics,
      timestamp: now,
      elapsed: now - this.startTime,
    };
    this.lastUpdate = now;
  }

  /**
   * Get current resource usage
   * @returns {Object} Resource metrics
   */
  async getResourceMetrics() {
    // Get TF.js memory info
    this.metrics.memory = await this.getTFMemoryInfo();

    // Add timing info
    this.metrics.device = {
      uptime: Date.now() - this.startTime,
      lastUpdate: this.lastUpdate,
    };

    return this.metrics;
  }

  /**
   * Check if memory usage is within safe limits
   * @returns {Object} Memory status
   */
  async checkMemoryStatus() {
    const memInfo = await this.getTFMemoryInfo();

    // Define thresholds (adjust based on device capabilities)
    const MEMORY_THRESHOLDS = {
      numTensors: 1000,
      bytesInMemory: 100 * 1024 * 1024, // 100MB
    };

    const status = {
      isHealthy: true,
      warnings: [],
    };

    if (memInfo.numTensors > MEMORY_THRESHOLDS.numTensors) {
      status.isHealthy = false;
      status.warnings.push(`High tensor count: ${memInfo.numTensors}`);
    }

    if (memInfo.bytesInMemory > MEMORY_THRESHOLDS.bytesInMemory) {
      status.isHealthy = false;
      status.warnings.push(
        `High memory usage: ${memInfo.bytesInMemory / 1024 / 1024}MB`,
      );
    }

    return status;
  }

  /**
   * Clean up unused tensors
   */
  async cleanup() {
    // Dispose of unused tensors
    tf.dispose();

    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
  }

  /**
   * Reset monitoring state
   */
  reset() {
    this.metrics = {
      memory: {},
      training: {},
      device: {},
    };
    this.startTime = Date.now();
    this.lastUpdate = this.startTime;
  }
}

export default new ResourceMonitor();
