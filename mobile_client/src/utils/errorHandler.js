/**
 * Error Handling and Logging Utilities for FedMob
 * Centralized error management and logging system
 */

class ErrorHandler {
  constructor() {
    this.logLevel = 'info'; // debug, info, warn, error
    this.logs = [];
    this.maxLogs = 1000;
    this.errorCount = 0;
    this.warningCount = 0;
  }

  /**
   * Set logging level
   * @param {string} level - Log level (debug, info, warn, error)
   */
  setLogLevel(level) {
    const validLevels = ['debug', 'info', 'warn', 'error'];
    if (validLevels.includes(level)) {
      this.logLevel = level;
      this.log('info', `Log level set to: ${level}`);
    } else {
      this.log('warn', `Invalid log level: ${level}. Using: ${this.logLevel}`);
    }
  }

  /**
   * Log a message with timestamp and level
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @param {Object} data - Additional data to log
   * @param {string} component - Component name
   */
  log(level, message, data = null, component = 'FedMob') {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      component,
      message,
      data,
    };

    // Add to logs array
    this.logs.push(logEntry);

    // Maintain max logs limit
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // Update counters
    if (level === 'error') {
      this.errorCount++;
    } else if (level === 'warn') {
      this.warningCount++;
    }

    // Console output based on level
    const shouldLog = this.shouldLog(level);
    if (shouldLog) {
      const logMessage = `[${timestamp}] [${level.toUpperCase()}] [${component}] ${message}`;

      switch (level) {
        case 'debug':
          console.debug(logMessage, data || '');
          break;
        case 'info':
          console.info(logMessage, data || '');
          break;
        case 'warn':
          console.warn(logMessage, data || '');
          break;
        case 'error':
          console.error(logMessage, data || '');
          break;
        default:
          console.log(logMessage, data || '');
      }
    }
  }

  /**
   * Check if message should be logged based on current level
   * @param {string} level - Message level
   * @returns {boolean} True if should log
   */
  shouldLog(level) {
    const levels = { debug: 0, info: 1, warn: 2, error: 3 };
    const currentLevel = levels[this.logLevel] || 1;
    const messageLevel = levels[level] || 1;
    return messageLevel >= currentLevel;
  }

  /**
   * Log debug message
   * @param {string} message - Debug message
   * @param {Object} data - Additional data
   * @param {string} component - Component name
   */
  debug(message, data = null, component = 'FedMob') {
    this.log('debug', message, data, component);
  }

  /**
   * Log info message
   * @param {string} message - Info message
   * @param {Object} data - Additional data
   * @param {string} component - Component name
   */
  info(message, data = null, component = 'FedMob') {
    this.log('info', message, data, component);
  }

  /**
   * Log warning message
   * @param {string} message - Warning message
   * @param {Object} data - Additional data
   * @param {string} component - Component name
   */
  warn(message, data = null, component = 'FedMob') {
    this.log('warn', message, data, component);
  }

  /**
   * Log error message
   * @param {string} message - Error message
   * @param {Object} data - Additional data
   * @param {string} component - Component name
   */
  error(message, data = null, component = 'FedMob') {
    this.log('error', message, data, component);
  }

  /**
   * Handle and log errors with context
   * @param {Error} error - Error object
   * @param {string} context - Error context
   * @param {Object} additionalData - Additional data
   * @param {string} component - Component name
   */
  handleError(
    error,
    context = 'Unknown',
    additionalData = null,
    component = 'FedMob',
  ) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      name: error.name,
      context,
      additionalData,
    };

    this.error(`Error in ${context}: ${error.message}`, errorInfo, component);

    // Return error info for further handling
    return errorInfo;
  }

  /**
   * Create a custom error with context
   * @param {string} message - Error message
   * @param {string} context - Error context
   * @param {Object} data - Additional data
   * @returns {Error} Custom error
   */
  createError(message, context = 'Unknown', data = null) {
    const error = new Error(message);
    error.context = context;
    error.data = data;
    error.timestamp = new Date().toISOString();

    this.error(`Custom error created: ${message}`, { context, data });
    return error;
  }

  /**
   * Wrap async function with error handling
   * @param {Function} asyncFn - Async function to wrap
   * @param {string} context - Error context
   * @param {string} component - Component name
   * @returns {Function} Wrapped function
   */
  wrapAsync(asyncFn, context = 'Async operation', component = 'FedMob') {
    return async (...args) => {
      try {
        this.debug(`Starting ${context}`, { args }, component);
        const result = await asyncFn(...args);
        this.debug(`Completed ${context}`, { result }, component);
        return result;
      } catch (error) {
        this.handleError(error, context, { args }, component);
        throw error;
      }
    };
  }

  /**
   * Wrap sync function with error handling
   * @param {Function} syncFn - Sync function to wrap
   * @param {string} context - Error context
   * @param {string} component - Component name
   * @returns {Function} Wrapped function
   */
  wrapSync(syncFn, context = 'Sync operation', component = 'FedMob') {
    return (...args) => {
      try {
        this.debug(`Starting ${context}`, { args }, component);
        const result = syncFn(...args);
        this.debug(`Completed ${context}`, { result }, component);
        return result;
      } catch (error) {
        this.handleError(error, context, { args }, component);
        throw error;
      }
    };
  }

  /**
   * Get error statistics
   * @returns {Object} Error statistics
   */
  getStats() {
    return {
      totalLogs: this.logs.length,
      errorCount: this.errorCount,
      warningCount: this.warningCount,
      logLevel: this.logLevel,
      maxLogs: this.maxLogs,
    };
  }

  /**
   * Get recent logs
   * @param {number} count - Number of recent logs to return
   * @param {string} level - Filter by log level
   * @returns {Array} Recent log entries
   */
  getRecentLogs(count = 50, level = null) {
    let filteredLogs = this.logs;

    if (level) {
      filteredLogs = this.logs.filter(log => log.level === level);
    }

    return filteredLogs.slice(-count);
  }

  /**
   * Get logs by component
   * @param {string} component - Component name
   * @returns {Array} Logs for component
   */
  getLogsByComponent(component) {
    return this.logs.filter(log => log.component === component);
  }

  /**
   * Clear all logs
   */
  clearLogs() {
    this.logs = [];
    this.errorCount = 0;
    this.warningCount = 0;
    this.info('Logs cleared');
  }

  /**
   * Export logs as JSON
   * @returns {string} JSON string of logs
   */
  exportLogs() {
    return JSON.stringify(this.logs, null, 2);
  }

  /**
   * Check if there are any errors
   * @returns {boolean} True if there are errors
   */
  hasErrors() {
    return this.errorCount > 0;
  }

  /**
   * Get last error
   * @returns {Object|null} Last error log entry
   */
  getLastError() {
    const errorLogs = this.logs.filter(log => log.level === 'error');
    return errorLogs.length > 0 ? errorLogs[errorLogs.length - 1] : null;
  }

  /**
   * Validate error handling setup
   * @returns {Object} Validation results
   */
  validateSetup() {
    const validation = {
      isValid: true,
      issues: [],
    };

    if (!this.logLevel) {
      validation.isValid = false;
      validation.issues.push('Log level not set');
    }

    if (this.maxLogs <= 0) {
      validation.isValid = false;
      validation.issues.push('Invalid max logs setting');
    }

    if (this.errorCount > 100) {
      validation.issues.push('High error count detected');
    }

    return validation;
  }
}

// Create singleton instance
const errorHandler = new ErrorHandler();

export default errorHandler;
export { ErrorHandler };
