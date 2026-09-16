// React Native asset linking — bundles the Ember brand fonts into both platforms.
// Run `npx react-native-asset` after `npm install` to copy the fonts into the iOS
// app target (Copy Bundle Resources) and Android (android/app/src/main/assets/fonts).
// The Android copies are also committed under android/app/src/main/assets/fonts so a
// plain `gradlew` build picks them up even without running the asset tool.
//
// Reference the fonts in JS by their internal family names (verified via fontTools):
//   sans/display → "Hanken Grotesk"   (variable, wght 100–900)
//   mono         → "JetBrains Mono"   (variable, wght 100–800)
module.exports = {
  project: {
    ios: {},
    android: {},
  },
  assets: ['./src/assets/fonts'],
};
