import {Platform} from 'react-native';
import RNFS from 'react-native-fs';

const MODEL_FILE_NAME = 'model_10m.pt';

// Search paths in priority order (allows pushing model directly to device)
const ANDROID_MODEL_SEARCH_PATHS = [
  () => `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`,       // already extracted
  () => `/sdcard/Download/${MODEL_FILE_NAME}`,                    // adb-pushed location
  () => `/sdcard/${MODEL_FILE_NAME}`,                             // fallback sdcard root
];

export async function ensureNativeModelPath() {
  if (Platform.OS === 'ios') {
    return `${RNFS.MainBundlePath}/${MODEL_FILE_NAME}`;
  }

  // Check if model was already pushed directly to the device
  for (const pathFn of ANDROID_MODEL_SEARCH_PATHS) {
    const p = pathFn();
    const exists = await RNFS.exists(p);
    if (exists) {
      // If found outside the app's files dir, copy it there for native C++ access
      const dest = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;
      if (p !== dest) {
        const destExists = await RNFS.exists(dest);
        if (destExists) await RNFS.unlink(dest);
        await RNFS.copyFile(p, dest);
        return dest;
      }
      return p;
    }
  }

  // Fallback: try APK assets (will fail gracefully if not bundled)
  try {
    const destPath = `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;
    const exists = await RNFS.exists(destPath);
    if (exists) await RNFS.unlink(destPath);
    await RNFS.copyFileAssets(MODEL_FILE_NAME, destPath);
    return destPath;
  } catch (e) {
    throw new Error(
      `Model ${MODEL_FILE_NAME} not found. Push it to /sdcard/Download/ with:\n` +
      `adb push ${MODEL_FILE_NAME} /sdcard/Download/`,
    );
  }
}

export {MODEL_FILE_NAME};
