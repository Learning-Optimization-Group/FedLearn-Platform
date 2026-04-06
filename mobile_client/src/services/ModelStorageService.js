/**
 * Model Storage Service for FedMob
 * Platform-specific storage service that handles web and mobile storage
 */

import PlatformStorageService from './PlatformStorageService';

// Re-export all methods from PlatformStorageService
export const saveModel = PlatformStorageService.saveModel.bind(
  PlatformStorageService,
);
export const loadModel = PlatformStorageService.loadModel.bind(
  PlatformStorageService,
);
export const deleteModel = PlatformStorageService.deleteModel.bind(
  PlatformStorageService,
);
export const listModels = PlatformStorageService.listModels.bind(
  PlatformStorageService,
);
export const getStorageInfo = PlatformStorageService.getStorageInfo.bind(
  PlatformStorageService,
);

// Legacy method names for backward compatibility
export const getAllModels = PlatformStorageService.listModels.bind(
  PlatformStorageService,
);

// Default export
export default PlatformStorageService;
