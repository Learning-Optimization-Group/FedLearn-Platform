// Babel config — React Native preset + NativeWind (Tailwind-for-RN) preset.
// react-native-worklets/plugin MUST be listed LAST. react-native-reanimated 4.x runs its worklets
// through this Babel plugin at build time; NativeWind's react-native-css-interop pulls reanimated in at
// render time, so without the plugin reanimated throws the moment the first styled view mounts — which
// silently blanks the whole app on a release build (no redbox to surface the error).
module.exports = {
  presets: ['module:@react-native/babel-preset', 'nativewind/babel'],
  plugins: ['react-native-worklets/plugin'],
};
