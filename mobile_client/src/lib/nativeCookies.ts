import networking from 'react-native/Libraries/Network/RCTNetworking';

/** An empty store reports false; an unavailable callback must fail startup rather than hang it. */
export function clearNativeCookies(): Promise<void> {
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => {
      if (finished) return;
      finished = true;
      reject(new Error('Could not clear the previous server session. Try again.'));
    }, 5000);
    try {
      networking.clearCookies(() => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        resolve();
      });
    } catch {
      finished = true;
      clearTimeout(timer);
      reject(new Error('Could not clear the previous server session. Try again.'));
    }
  });
}
