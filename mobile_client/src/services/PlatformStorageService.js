import { Platform } from 'react-native';
import RNFS from 'react-native-fs';

class PlatformStorageService {
  constructor() {
    this.isWeb = Platform.OS === 'web';
    this.webCache = new Map(); // In-memory cache for web
    
    // Storage configuration
    this.config = {
      maxModels: 10, // Keep only last 10 models
      maxStorageMB: 50, // Alert when approaching 50MB
      warningThresholdMB: 40, // Warn at 40MB
      emergencyCleanupModels: 5, // In emergency, keep only 5 models
    };
  }

  async saveModel(modelData) {
    if (this.isWeb) {
      return this.saveModelWeb(modelData);
    } else {
      return this.saveModelMobile(modelData);
    }
  }

  async loadModel(modelId) {
    if (this.isWeb) {
      return this.loadModelWeb(modelId);
    } else {
      return this.loadModelMobile(modelId);
    }
  }

  async deleteModel(modelId) {
    if (this.isWeb) {
      return this.deleteModelWeb(modelId);
    } else {
      return this.deleteModelMobile(modelId);
    }
  }

  async listModels() {
    if (this.isWeb) {
      return this.listModelsWeb();
    } else {
      return this.listModelsMobile();
    }
  }

  async getStorageInfo() {
    if (this.isWeb) {
      return this.getStorageInfoWeb();
    } else {
      return this.getStorageInfoMobile();
    }
  }

  async getStorageInfoWeb() {
    try {
      const models = this.getStoredModels();
      const totalSizeBytes = Object.values(models).reduce((total, model) => {
        return total + JSON.stringify(model).length;
      }, 0);

      return {
        totalModels: Object.keys(models).length,
        totalSizeBytes: totalSizeBytes,
        totalSizeMB: (totalSizeBytes / 1024 / 1024).toFixed(2),
      };
    } catch (error) {
      console.error('Failed to get web storage info:', error);
      return { totalModels: 0, totalSizeBytes: 0, totalSizeMB: '0.00' };
    }
  }

  async getStorageInfoMobile() {
    try {
      const models = await this.listModelsMobile();
      const totalSizeBytes = this._totalSizeBytes || 0;

      return {
        totalModels: models.length,
        totalSizeBytes: totalSizeBytes,
        totalSizeMB: (totalSizeBytes / 1024 / 1024).toFixed(2),
      };
    } catch (error) {
      console.error('Failed to get storage info:', error);
      return { totalModels: 0, totalSizeBytes: 0, totalSizeMB: '0.00' };
    }
  }

  // Web storage methods (using in-memory cache)
  async saveModelWeb(modelData) {
    try {
      const modelId = modelData.id || `model_${Date.now()}`;
      const modelToSave = {
        ...modelData,
        id: modelId,
        savedAt: new Date().toISOString(),
      };
      
      // Check storage before saving
      await this.checkAndCleanupStorage();
      
      this.webCache.set(modelId, modelToSave);

      // Try to save to localStorage with error handling
      try {
        const models = this.getStoredModels();
        models[modelId] = modelToSave;
        localStorage.setItem('fedmob_models', JSON.stringify(models));
        console.log(`Model saved to web cache: ${modelId}`);
      } catch (quotaError) {
        if (quotaError.name === 'QuotaExceededError') {
          console.warn('Storage quota exceeded, performing emergency cleanup...');
          await this.emergencyCleanup();
          
          // Retry saving after cleanup
          const models = this.getStoredModels();
          models[modelId] = modelToSave;
          localStorage.setItem('fedmob_models', JSON.stringify(models));
          console.log(`Model saved to web cache after emergency cleanup: ${modelId}`);
        } else {
          throw quotaError;
        }
      }
      
      return modelId;
    } catch (error) {
      console.error('Failed to save model to web cache:', error);
      throw error;
    }
  }

  async loadModelWeb(modelId) {
    try {
      // Try cache first
      if (this.webCache.has(modelId)) {
        return this.webCache.get(modelId);
      }

      // Try localStorage
      const models = this.getStoredModels();
      if (models[modelId]) {
        this.webCache.set(modelId, models[modelId]);
        return models[modelId];
      }

      throw new Error(`Model not found: ${modelId}`);
    } catch (error) {
      console.error('Failed to load model from web cache:', error);
      throw error;
    }
  }

  async deleteModelWeb(modelId) {
    try {
      this.webCache.delete(modelId);

      // Also remove from localStorage
      const models = this.getStoredModels();
      delete models[modelId];
      localStorage.setItem('fedmob_models', JSON.stringify(models));

      console.log(`Model deleted from web cache: ${modelId}`);
    } catch (error) {
      console.error('Failed to delete model from web cache:', error);
      throw error;
    }
  }

  async listModelsWeb() {
    try {
      const models = this.getStoredModels();
      return Object.values(models).map(model => ({
        id: model.id,
        name: model.name,
        timestamp: model.savedAt, // Map savedAt to timestamp for compatibility
        savedAt: model.savedAt,
        accuracy: model.accuracy,
        round: model.round,
        metrics: model.metrics, // Include metrics if available
      }));
    } catch (error) {
      console.error('Failed to list models from web cache:', error);
      return [];
    }
  }

  getStoredModels() {
    try {
      const stored = localStorage.getItem('fedmob_models');
      return stored ? JSON.parse(stored) : {};
    } catch (error) {
      console.warn('Failed to parse stored models:', error);
      return {};
    }
  }

  // Storage management methods
  async checkStorageSize() {
    const storageInfo = await this.getStorageInfoWeb();
    const sizeMB = parseFloat(storageInfo.totalSizeMB);
    
    console.log(`Current storage: ${sizeMB}MB (${storageInfo.totalModels} models)`);
    
    if (sizeMB > this.config.warningThresholdMB) {
      console.warn(`Storage approaching limit: ${sizeMB}MB / ${this.config.maxStorageMB}MB`);
    }
    
    return {
      sizeMB,
      isNearLimit: sizeMB > this.config.warningThresholdMB,
      exceedsLimit: sizeMB > this.config.maxStorageMB,
      modelCount: storageInfo.totalModels
    };
  }

  async checkAndCleanupStorage() {
    const storageStatus = await this.checkStorageSize();
    
    // Always cleanup if we have too many models
    if (storageStatus.modelCount > this.config.maxModels) {
      console.log(`Too many models (${storageStatus.modelCount}), cleaning up...`);
      await this.cleanupOldModels(this.config.maxModels);
    }
    
    // Also cleanup if approaching size limit
    if (storageStatus.exceedsLimit || storageStatus.isNearLimit) {
      console.log(`Storage size limit approached (${storageStatus.sizeMB}MB), cleaning up...`);
      await this.cleanupOldModels(Math.max(5, this.config.maxModels - 3));
    }
  }

  async cleanupOldModels(keepCount = this.config.maxModels) {
    try {
      const models = this.getStoredModels();
      const modelList = Object.values(models).sort((a, b) => 
        new Date(b.savedAt) - new Date(a.savedAt)
      );
      
      if (modelList.length <= keepCount) {
        console.log(`No cleanup needed: ${modelList.length} models <= ${keepCount} limit`);
        return;
      }
      
      const modelsToKeep = modelList.slice(0, keepCount);
      const modelsToDelete = modelList.slice(keepCount);
      
      console.log(`Cleaning up ${modelsToDelete.length} old models, keeping ${modelsToKeep.length}`);
      
      // Create new models object with only recent models
      const cleanedModels = {};
      modelsToKeep.forEach(model => {
        cleanedModels[model.id] = model;
        // Keep in cache too
        this.webCache.set(model.id, model);
      });
      
      // Remove old models from cache
      modelsToDelete.forEach(model => {
        this.webCache.delete(model.id);
        console.log(`Removed old model: ${model.id} (saved: ${model.savedAt})`);
      });
      
      // Update localStorage
      localStorage.setItem('fedmob_models', JSON.stringify(cleanedModels));
      
      const newStorageInfo = await this.getStorageInfoWeb();
      console.log(`Cleanup completed: ${newStorageInfo.totalModels} models, ${newStorageInfo.totalSizeMB}MB`);
      
    } catch (error) {
      console.error('Failed to cleanup old models:', error);
      throw error;
    }
  }

  async emergencyCleanup() {
    console.warn('Performing emergency cleanup due to storage quota exceeded');
    try {
      // Keep only the most recent models in emergency
      await this.cleanupOldModels(this.config.emergencyCleanupModels);
      
      // If still having issues, clear everything except the newest model
      const storageStatus = await this.checkStorageSize();
      if (storageStatus.exceedsLimit) {
        console.warn('Emergency cleanup: keeping only 1 most recent model');
        await this.cleanupOldModels(1);
      }
    } catch (error) {
      console.error('Emergency cleanup failed:', error);
      // Last resort: clear all models
      console.warn('Last resort: clearing all stored models');
      localStorage.removeItem('fedmob_models');
      this.webCache.clear();
    }
  }

  // Manual cleanup methods for UI
  async clearAllModels() {
    try {
      if (this.isWeb) {
        localStorage.removeItem('fedmob_models');
        this.webCache.clear();
        console.log('All web models cleared');
      } else {
        const models = await this.listModelsMobile();
        for (const model of models) {
          await this.deleteModelMobile(model.id);
        }
        console.log('All mobile models cleared');
      }
    } catch (error) {
      console.error('Failed to clear all models:', error);
      throw error;
    }
  }

  async getStorageQuotaInfo() {
    if (!this.isWeb) {
      return { supported: false, message: 'Storage quota API not available on mobile' };
    }
    
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        const quotaMB = estimate.quota ? (estimate.quota / 1024 / 1024).toFixed(2) : 'Unknown';
        const usageMB = estimate.usage ? (estimate.usage / 1024 / 1024).toFixed(2) : 'Unknown';
        
        return {
          supported: true,
          quotaMB,
          usageMB,
          availableMB: estimate.quota && estimate.usage ? 
            ((estimate.quota - estimate.usage) / 1024 / 1024).toFixed(2) : 'Unknown'
        };
      }
    } catch (error) {
      console.warn('Failed to get storage quota info:', error);
    }
    
    return { supported: false, message: 'Storage quota API not supported' };
  }

  // Mobile storage methods (using react-native-fs)
  async saveModelMobile(modelData) {
    try {
      const modelId = modelData.id || `model_${Date.now()}`;
      const fileName = `model_${modelId}.json`;
      const filePath = `${RNFS.DocumentDirectoryPath}/${fileName}`;

      const modelToSave = {
        ...modelData,
        id: modelId,
        savedAt: new Date().toISOString(),
      };

      await RNFS.writeFile(
        filePath,
        JSON.stringify(modelToSave, null, 2),
        'utf8',
      );

      console.log(`Model saved to mobile storage: ${modelId}`);
      return modelId;
    } catch (error) {
      console.error('Failed to save model to mobile storage:', error);
      throw error;
    }
  }

  async loadModelMobile(modelId) {
    try {
      const fileName = `model_${modelId}.json`;
      const filePath = `${RNFS.DocumentDirectoryPath}/${fileName}`;

      const fileExists = await RNFS.exists(filePath);
      if (!fileExists) {
        throw new Error(`Model file not found: ${modelId}`);
      }

      const modelData = await RNFS.readFile(filePath, 'utf8');
      return JSON.parse(modelData);
    } catch (error) {
      console.error('Failed to load model from mobile storage:', error);
      throw error;
    }
  }

  async deleteModelMobile(modelId) {
    try {
      const fileName = `model_${modelId}.json`;
      const filePath = `${RNFS.DocumentDirectoryPath}/${fileName}`;

      const fileExists = await RNFS.exists(filePath);
      if (fileExists) {
        await RNFS.unlink(filePath);
        console.log(`Model deleted from mobile storage: ${modelId}`);
      }
    } catch (error) {
      console.error('Failed to delete model from mobile storage:', error);
      throw error;
    }
  }

  async listModelsMobile() {
    try {
      const files = await RNFS.readDir(RNFS.DocumentDirectoryPath);
      const modelFiles = files.filter(
        file => file.name.startsWith('model_') && file.name.endsWith('.json'),
      );

      const models = [];
      let totalSizeBytes = 0;

      for (const file of modelFiles) {
        try {
          const modelData = await RNFS.readFile(file.path, 'utf8');
          const model = JSON.parse(modelData);

          // Calculate file size
          const fileSize = file.size || modelData.length;
          totalSizeBytes += fileSize;

          models.push({
            id: model.id,
            name: model.name,
            timestamp: model.savedAt, // Map savedAt to timestamp for compatibility
            savedAt: model.savedAt,
            accuracy: model.accuracy,
            round: model.round,
            metrics: model.metrics, // Include metrics if available
            size: fileSize,
          });
        } catch (error) {
          console.warn(`Failed to parse model file ${file.name}:`, error);
        }
      }

      // Store total size for getStorageInfo
      this._totalSizeBytes = totalSizeBytes;

      return models.sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
    } catch (error) {
      console.error('Failed to list models from mobile storage:', error);
      return [];
    }
  }
}

export default new PlatformStorageService();
