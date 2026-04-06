// src/screens/TrainingScreen.jsx

import React, {useState, useEffect, useRef, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Alert,
  StatusBar,
  SafeAreaView,
  Platform,
} from 'react-native';
import {NativeModules} from 'react-native';
import {ensureNativeModelPath, MODEL_FILE_NAME} from '../utils/nativeModelPath';
import PlatformStorageService from '../services/PlatformStorageService';

const NativeFedLearnCore =
  NativeModules.NativeFedLearnCore ||
  (typeof global !== 'undefined' && global.__turboModuleProxy
    ? global.__turboModuleProxy('NativeFedLearnCore')
    : null);

console.log('[FL] TrainingScreen module loaded, NativeFedLearnCore:', !!NativeFedLearnCore);

const TRAINING_MODES = [
  {key: 'fedavg',  label: 'FedAvg (SGD)'},
  {key: 'zo_fl',   label: 'ZO-FL (FedAvg + ZO)', disabled: true},
  {key: 'decomfl', label: 'DeComFL',              disabled: true},
];

const TrainingScreen = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [serverAddress, setServerAddress] = useState('localhost:50063');
  const [clientId, setClientId] = useState(`mobile_${Date.now()}`);
  const [currentRound, setCurrentRound] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [loss, setLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [trainingMode, setTrainingMode] = useState('fedavg');
  const [errorMsg, setErrorMsg] = useState('');
  const statusPollRef = useRef(null);
  const modelPathRef = useRef(null);
  const modelInfoRef = useRef(null);
  const lastSavedRoundRef = useRef(-1);

  const addLog = useCallback(message => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-49), `[${timestamp}] ${message}`]);
  }, []);

  const saveModelToLibrary = useCallback(async (round, latestAccuracy, latestLoss) => {
    if (!modelPathRef.current || !modelInfoRef.current) return;
    try {
      await PlatformStorageService.saveModel({
        id: `trained_${MODEL_FILE_NAME.replace('.pt', '')}`,
        name: round > 0
          ? `${MODEL_FILE_NAME} (Round ${round})`
          : `${MODEL_FILE_NAME} (baseline)`,
        accuracy: latestAccuracy > 0 ? latestAccuracy : null,
        loss: latestLoss > 0 ? latestLoss : null,
        round,
        ptPath: modelPathRef.current,
        numParams: modelInfoRef.current.numParams,
        sizeBytes: modelInfoRef.current.sizeBytes,
      });
      console.log('[TrainingScreen] Model saved to library, round:', round);
    } catch (e) {
      console.log('[TrainingScreen] saveModelToLibrary error:', e.message);
    }
  }, []);

  useEffect(() => {
    const init = async () => {
      console.log('[TrainingScreen] NativeFedLearnCore =', NativeFedLearnCore);
      if (!NativeFedLearnCore) {
        console.log('[TrainingScreen] Module NOT available');
        addLog('NativeFedLearnCore module not available (build required)');
        setModelReady(false);
        return;
      }
      console.log('[TrainingScreen] Module IS available, loading model...');

      try {
        addLog('Loading TorchScript model...');
        const modelPath = await ensureNativeModelPath();
        console.log('[TrainingScreen] Resolved model path:', modelPath);

        const loaded = await NativeFedLearnCore.loadModel(modelPath);
        console.log('[TrainingScreen] loadModel returned:', loaded);
        if (loaded) {
          const infoStr = await NativeFedLearnCore.getModelInfo();
          console.log('[TrainingScreen] getModelInfo:', infoStr);
          const info = JSON.parse(infoStr);
          setModelInfo(info);
          setModelReady(true);
          // Store refs for library save
          modelPathRef.current = modelPath;
          modelInfoRef.current = info;
          lastSavedRoundRef.current = -1;
          // Register model in library immediately (baseline entry)
          await saveModelToLibrary(0, 0, 0);
          addLog(
            `Model loaded (${MODEL_FILE_NAME}): ${info.numParams.toLocaleString()} params (${(info.sizeBytes / 1024 / 1024).toFixed(2)} MB)`,
          );
        } else {
          console.log('[TrainingScreen] loadModel returned false');
          addLog('Failed to load model');
        }
      } catch (error) {
        console.log('[TrainingScreen] init error:', error?.message);
        addLog(`Init error: ${error.message}`);
      }
    };

    init();

    return () => {
      if (statusPollRef.current) clearInterval(statusPollRef.current);
    };
  }, [addLog]);

  const startStatusPolling = useCallback(() => {
    if (statusPollRef.current) clearInterval(statusPollRef.current);

    statusPollRef.current = setInterval(async () => {
      try {
        const statusStr = await NativeFedLearnCore.getStatus();
        const status = JSON.parse(statusStr);

        setTrainingStatus(status.phase);
        if (status.round > 0) setCurrentRound(status.round);
        if (status.loss > 0) setLoss(status.loss);
        if (status.accuracy > 0) setAccuracy(status.accuracy);
        if (status.totalSteps > 0) {
          setProgress(
            Math.round((status.step / status.totalSteps) * 100),
          );
        }
        if (status.error) setErrorMsg(status.error);

        // Save model to library whenever a new round completes
        if (status.round > 0 && status.round !== lastSavedRoundRef.current) {
          lastSavedRoundRef.current = status.round;
          await saveModelToLibrary(status.round, status.accuracy, status.loss);
        }

        if (status.phase === 'stopped' || status.phase === 'error') {
          // Final save with latest metrics on completion
          if (status.round > 0) {
            await saveModelToLibrary(status.round, status.accuracy, status.loss);
          }
          clearInterval(statusPollRef.current);
          statusPollRef.current = null;
        }
      } catch {
        // polling error, ignore
      }
    }, 2000);
  }, []);

  const handleConnect = async () => {
    if (!NativeFedLearnCore) {
      Alert.alert('Error', 'Native module not available. Build with C++ support required.');
      return;
    }

    try {
      addLog(`Connecting to ${serverAddress}...`);
      setTrainingStatus('connecting');
      const success = await NativeFedLearnCore.connect(serverAddress, clientId);
      if (success) {
        setIsConnected(true);
        setTrainingStatus('connected');
        addLog('Connected via native gRPC');
      } else {
        setTrainingStatus('error');
        addLog('Connection failed');
        Alert.alert('Connection Failed', 'Could not connect to server.');
      }
    } catch (error) {
      setTrainingStatus('error');
      addLog(`Connection error: ${error.message}`);
    }
  };

  const handleDisconnect = async () => {
    try {
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
      await NativeFedLearnCore.disconnect();
      setIsConnected(false);
      setTrainingStatus('idle');
      setCurrentRound(0);
      setLoss(0);
      setAccuracy(0);
      setProgress(0);
      addLog('Disconnected');
    } catch (error) {
      addLog(`Disconnect error: ${error.message}`);
    }
  };

  const handleLocalTrain = async () => {
    try {
      addLog('Starting local training (no server)...');
      setTrainingStatus('training');
      const resultStr = await NativeFedLearnCore.trainStep('data', 3, 0.01);
      const result = JSON.parse(resultStr);
      if (result.error) {
        addLog(`Train error: ${result.error}`);
        setTrainingStatus('error');
      } else {
        setLoss(result.loss);
        setAccuracy(result.accuracy);
        setTrainingStatus('completed');
        addLog(
          `Local training done: loss=${result.loss.toFixed(4)}, accuracy=${(result.accuracy * 100).toFixed(1)}%`,
        );
      }
    } catch (error) {
      setTrainingStatus('error');
      addLog(`Local train error: ${error.message}`);
    }
  };

  const handleStartFL = async () => {
    try {
      const config = JSON.stringify({
        local_epochs: 2,
        learning_rate: 0.01,
        batch_size: 32,
      });

      if (trainingMode === 'decomfl') {
        addLog('Starting DeComFL training...');
        await NativeFedLearnCore.startDeComFLLoop(config);
      } else if (trainingMode === 'zo_fl') {
        addLog('Starting ZO-FL training...');
        await NativeFedLearnCore.setZOConfig(
          JSON.stringify({mu: 0.001, numPert: 10}),
        );
        await NativeFedLearnCore.startFedAvgLoop(config);
      } else {
        addLog('Starting FedAvg training...');
        await NativeFedLearnCore.startFedAvgLoop(config);
      }

      setTrainingStatus('training');
      startStatusPolling();
    } catch (error) {
      setTrainingStatus('error');
      addLog(`FL start error: ${error.message}`);
    }
  };

  const handleStopTraining = async () => {
    try {
      await NativeFedLearnCore.stopTraining();
      if (statusPollRef.current) {
        clearInterval(statusPollRef.current);
        statusPollRef.current = null;
      }
      setTrainingStatus('stopped');
      addLog('Training stopped');
    } catch (error) {
      addLog(`Stop error: ${error.message}`);
    }
  };

  const isTraining =
    trainingStatus === 'training' ||
    trainingStatus === 'fetching' ||
    trainingStatus === 'uploading' ||
    trainingStatus === 'registering';

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />

      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>FedMob</Text>
          <Text style={styles.subtitle}>
            Federated Learning Mobile Client (C++ Core)
          </Text>
        </View>

        {/* Connection Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Connection</Text>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Server Address (gRPC):</Text>
            <TextInput
              style={styles.input}
              value={serverAddress}
              onChangeText={setServerAddress}
              placeholder="10.5.1.254:50051"
              editable={!isConnected}
            />
          </View>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Client ID:</Text>
            <TextInput
              style={styles.input}
              value={clientId}
              onChangeText={setClientId}
              placeholder="mobile_client_123"
              editable={!isConnected}
            />
          </View>

          <View style={styles.buttonContainer}>
            {!isConnected ? (
              <TouchableOpacity
                style={[
                  styles.connectButton,
                  !modelReady && styles.disabledButton,
                ]}
                onPress={handleConnect}
                disabled={!modelReady}>
                <Text style={styles.buttonText}>
                  {modelReady ? 'Connect (gRPC)' : 'Loading model...'}
                </Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={styles.disconnectButton}
                onPress={handleDisconnect}>
                <Text style={styles.buttonText}>Disconnect</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Status Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Status</Text>

          <View style={styles.statusContainer}>
            <StatusRow
              label="Connection"
              value={isConnected ? 'Connected' : 'Disconnected'}
              color={isConnected ? '#28a745' : '#dc3545'}
            />
            <StatusRow
              label="Model"
              value={
                modelReady
                  ? `Ready (${modelInfo?.numParams?.toLocaleString() || '?'} params)`
                  : 'Loading...'
              }
              color={modelReady ? '#28a745' : '#ffc107'}
            />
            <StatusRow label="Phase" value={trainingStatus} />
            <StatusRow label="Round" value={String(currentRound)} />
            {loss > 0 && (
              <StatusRow label="Loss" value={loss.toFixed(4)} />
            )}
            {accuracy > 0 && (
              <StatusRow
                label="Accuracy"
                value={`${(accuracy * 100).toFixed(1)}%`}
              />
            )}
            {errorMsg ? (
              <StatusRow label="Error" value={errorMsg} color="#dc3545" />
            ) : null}
          </View>
        </View>

        {/* Training Mode Selector */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Training Mode</Text>
          <View style={styles.modeContainer}>
            {TRAINING_MODES.map(mode => (
              <TouchableOpacity
                key={mode.key}
                style={[
                  styles.modeButton,
                  trainingMode === mode.key && styles.modeButtonActive,
                  (isTraining || mode.disabled) && styles.disabledButton,
                ]}
                onPress={() => !mode.disabled && setTrainingMode(mode.key)}
                disabled={isTraining || mode.disabled}>
                <Text
                  style={[
                    styles.modeButtonText,
                    trainingMode === mode.key && styles.modeButtonTextActive,
                  ]}>
                  {mode.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.modeDescription}>
            {trainingMode === 'fedavg' &&
              'Standard FedAvg with SGD local training. Full model parameters are exchanged.'}
            {trainingMode === 'zo_fl' &&
              'Zeroth-Order optimization with FedAvg. Gradient-free training using forward passes only.'}
            {trainingMode === 'decomfl' &&
              'DeComFL: Decomposed FL with ZO gradient scalars. Byzantine-robust, communication-efficient.'}
          </Text>
        </View>

        {/* Training Controls */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Training</Text>

          {/* Local training (no server needed) */}
          {!isConnected && modelReady && (
            <View style={styles.buttonContainer}>
              <TouchableOpacity
                style={[
                  styles.localTrainButton,
                  isTraining && styles.disabledButton,
                ]}
                onPress={handleLocalTrain}
                disabled={isTraining}>
                <Text style={styles.buttonText}>Local Train (Demo)</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Federated training (server needed) */}
          {isConnected && (
            <View style={styles.buttonContainer}>
              {!isTraining ? (
                <TouchableOpacity
                  style={styles.flButton}
                  onPress={handleStartFL}>
                  <Text style={styles.buttonText}>
                    Start{' '}
                    {
                      TRAINING_MODES.find(m => m.key === trainingMode)
                        ?.label
                    }
                  </Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={styles.stopButton}
                  onPress={handleStopTraining}>
                  <Text style={styles.buttonText}>Stop Training</Text>
                </TouchableOpacity>
              )}
            </View>
          )}

          {/* Progress bar */}
          {isTraining && (
            <View style={styles.progressContainer}>
              <View style={[styles.progressBar, {width: `${progress}%`}]} />
              <Text style={styles.progressText}>{progress}%</Text>
            </View>
          )}
        </View>

        {/* Logs Section */}
        <View style={styles.section}>
          <View style={styles.logHeader}>
            <Text style={styles.sectionTitle}>Logs</Text>
            <TouchableOpacity onPress={() => setLogs([])}>
              <Text style={styles.clearLogs}>Clear</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.logsContainer}>
            {logs.slice(-15).map((log, index) => (
              <Text key={index} style={styles.logText}>
                {log}
              </Text>
            ))}
            {logs.length === 0 && (
              <Text style={styles.logPlaceholder}>No logs yet</Text>
            )}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const StatusRow = ({label, value, color}) => (
  <View style={styles.statusRow}>
    <Text style={styles.statusLabel}>{label}:</Text>
    <Text style={[styles.statusValue, color && {color}]}>{value}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  scrollView: {
    flex: 1,
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
    paddingTop: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 14,
    color: '#7f8c8d',
    textAlign: 'center',
  },
  section: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15,
  },
  inputContainer: {
    marginBottom: 15,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#34495e',
    marginBottom: 5,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#f8f9fa',
  },
  buttonContainer: {
    marginTop: 10,
  },
  connectButton: {
    backgroundColor: '#28a745',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  disconnectButton: {
    backgroundColor: '#dc3545',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  flButton: {
    backgroundColor: '#007bff',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  localTrainButton: {
    backgroundColor: '#6f42c1',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  stopButton: {
    backgroundColor: '#dc3545',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#6c757d',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  statusContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 15,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  statusLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#34495e',
  },
  statusValue: {
    fontSize: 14,
    color: '#2c3e50',
  },
  modeContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  modeButton: {
    flex: 1,
    padding: 10,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#dee2e6',
    alignItems: 'center',
  },
  modeButtonActive: {
    borderColor: '#007bff',
    backgroundColor: '#e7f1ff',
  },
  modeButtonText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#6c757d',
    textAlign: 'center',
  },
  modeButtonTextActive: {
    color: '#007bff',
  },
  modeDescription: {
    fontSize: 13,
    color: '#6c757d',
    fontStyle: 'italic',
  },
  progressContainer: {
    height: 20,
    backgroundColor: '#f0f0f0',
    borderRadius: 10,
    marginTop: 15,
    overflow: 'hidden',
    position: 'relative',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#28a745',
    borderRadius: 10,
  },
  progressText: {
    position: 'absolute',
    width: '100%',
    textAlign: 'center',
    color: '#000',
    fontSize: 12,
    lineHeight: 20,
  },
  logHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  clearLogs: {
    color: '#007bff',
    fontSize: 14,
  },
  logsContainer: {
    backgroundColor: '#1e1e1e',
    borderRadius: 8,
    padding: 15,
    maxHeight: 250,
  },
  logText: {
    fontSize: 11,
    color: '#d4d4d4',
    marginBottom: 3,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  logPlaceholder: {
    fontSize: 12,
    color: '#6c757d',
    fontStyle: 'italic',
  },
});

export default TrainingScreen;
