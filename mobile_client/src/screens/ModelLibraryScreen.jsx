// src/screens/ModelLibraryScreen.jsx

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import PlatformStorageService from '../services/PlatformStorageService';

const ModelLibraryScreen = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const navigation = useNavigation();

  // Load models on component mount
  useEffect(() => {
    loadModels();
  }, []);

  // Load models from storage
  const loadModels = async () => {
    try {
      setLoading(true);
      console.log('📚 Loading models from storage...');

      const savedModels = await PlatformStorageService.listModelsMobile();
      console.log(`📚 Found ${savedModels.length} models:`, savedModels);

      setModels(savedModels);
    } catch (error) {
      console.error('❌ Failed to load models:', error);
      Alert.alert('Error', 'Failed to load saved models');
    } finally {
      setLoading(false);
    }
  };

  // Refresh models
  const onRefresh = async () => {
    setRefreshing(true);
    await loadModels();
    setRefreshing(false);
  };

  // Test a model (navigate to inference)
  const testModel = model => {
    navigation.navigate('Inference', { selectedModel: model });
  };

  // Delete a model
  const deleteModel = async (modelId, modelName) => {
    Alert.alert(
      'Delete Model',
      `Are you sure you want to delete "${modelName}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await PlatformStorageService.deleteModelMobile(modelId);
              await loadModels(); // Reload the list
              Alert.alert('Success', 'Model deleted successfully');
            } catch (error) {
              Alert.alert('Error', 'Failed to delete model');
            }
          },
        },
      ],
    );
  };

  // Format file size
  const formatFileSize = bytes => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format date
  const formatDate = timestamp => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007bff" />
        <Text style={styles.loadingText}>Loading models...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.title}>Model Library</Text>
        <Text style={styles.subtitle}>
          {models.length} saved model{models.length !== 1 ? 's' : ''}
        </Text>
      </View>

      {models.length === 0 ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📚 Available Models</Text>
          <Text style={styles.emptyText}>No models available yet</Text>
          <Text style={styles.emptySubtext}>
            Train a model in the Training tab to see it here
          </Text>
        </View>
      ) : (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📚 Available Models</Text>
          {models.map(model => (
            <View key={model.id} style={styles.modelCard}>
              <View style={styles.modelHeader}>
                <Text style={styles.modelName}>{model.name}</Text>
                <Text style={styles.modelDate}>
                  {formatDate(model.savedAt || model.timestamp)}
                </Text>
              </View>

              <View style={styles.modelDetails}>
                {model.accuracy && (
                  <Text style={styles.modelMetric}>
                    Accuracy: {(model.accuracy * 100).toFixed(2)}%
                  </Text>
                )}
                {model.round && (
                  <Text style={styles.modelMetric}>Round: {model.round}</Text>
                )}
                <Text style={styles.modelMetric}>
                  Size: {formatFileSize(model.size)}
                </Text>
              </View>

              <View style={styles.modelActions}>
                <TouchableOpacity
                  style={styles.testButton}
                  onPress={() => testModel(model)}
                >
                  <Text style={styles.testButtonText}>🔍 Test Model</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.deleteButton}
                  onPress={() => deleteModel(model.id, model.name)}
                >
                  <Text style={styles.deleteButtonText}>🗑️ Delete</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 20,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#6c757d',
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
    color: '#7f8c8d',
  },
  section: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  emptyText: {
    textAlign: 'center',
    color: '#6c757d',
    fontStyle: 'italic',
    fontSize: 16,
    marginBottom: 10,
  },
  emptySubtext: {
    textAlign: 'center',
    color: '#adb5bd',
    fontSize: 14,
  },
  modelCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 15,
    marginBottom: 15,
    borderLeftWidth: 4,
    borderLeftColor: '#007bff',
  },
  modelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  modelName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2c3e50',
    flex: 1,
    marginRight: 10,
  },
  modelDate: {
    fontSize: 12,
    color: '#6c757d',
  },
  modelDetails: {
    marginBottom: 15,
  },
  modelMetric: {
    fontSize: 14,
    color: '#495057',
    marginBottom: 3,
  },
  modelActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  testButton: {
    backgroundColor: '#28a745',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 6,
    flex: 1,
    marginRight: 10,
  },
  testButtonText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  deleteButton: {
    backgroundColor: '#dc3545',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 6,
    flex: 1,
  },
  deleteButtonText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

export default ModelLibraryScreen;
