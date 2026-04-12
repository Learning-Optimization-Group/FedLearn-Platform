import {Platform} from 'react-native';
import RNFS from 'react-native-fs';

const MODEL_FILE_NAME = 'model_10m.pt';

export async function ensureNativeModelPath() {
  if (Platform.OS === 'ios') {
    return `${RNFS.MainBundlePath}/${MODEL_FILE_NAME}`;
  }

  const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;

  // Check app's private files directory (populated via adb push + run-as cp)
  try {
    const exists = await RNFS.exists(dest);
    if (exists) {
      return dest;
    }
  } catch (_) {}

  // Fallback: try APK assets (will fail gracefully if not bundled)
  try {
    const exists = await RNFS.exists(dest);
    if (exists) {
      await RNFS.unlink(dest);
    }
    await RNFS.copyFileAssets(MODEL_FILE_NAME, dest);
    return dest;
  } catch (e) {
    throw new Error(
      `Model ${MODEL_FILE_NAME} not found.\n` +
        `Push it with:\n` +
        `  adb push assets/${MODEL_FILE_NAME} /data/local/tmp/${MODEL_FILE_NAME}\n` +
        `  adb shell run-as com.mobileclientnew cp /data/local/tmp/${MODEL_FILE_NAME} files/${MODEL_FILE_NAME}`,
    );
  }
}

export {MODEL_FILE_NAME};
