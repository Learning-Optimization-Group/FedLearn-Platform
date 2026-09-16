#!/usr/bin/env bash
#
# Generate FedLearn.xcodeproj / FedLearn.xcworkspace for the React Native 0.80 (New Architecture)
# iOS app.
#
# HOW: RN 0.80 no longer ships its template inside the `react-native` package — it lives in the
# `@react-native-community/template` package, scaffolded by the community CLI. This script inits a
# throwaway project *named FedLearn* at the EXACT RN version pinned in package.json (so the generated
# pbxproj always matches that RN line and no fragile renaming is needed), copies its iOS Xcode project
# into ios/, overlays the support files this repo doesn't already have, and runs `pod install`.
# Only the declarative inputs (this script, ios/FedLearn/*, ios/Podfile) are committed; the pbxproj
# is reproducible.
#
# REQUIREMENTS — must run on a Mac (none of this is available/verifiable in CI):
#   - macOS with full Xcode (not just Command Line Tools)   ->  xcodebuild -version
#   - CocoaPods                                             ->  pod --version
#         install:  brew install cocoapods   (or)  sudo gem install cocoapods
#   - Node + network (npx fetches the community CLI + template)
#
# VERIFY-BEFORE-USE: the New-Architecture native wiring of the C++ FL core (the bridge .mm/.cpp, the
# Swift bridging header, the cross-compiled libtorch/gRPC xcframeworks) is RN-version / Xcode-project
# specific and is NOT auto-added to the target by this script — see "REMAINING" at the end. CLI flag
# names below are community-CLI-version sensitive; `npx @react-native-community/cli init --help`.
set -euo pipefail

IOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$IOS_DIR/.." && pwd)"
APP="FedLearn"

err(){  printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
ok(){   printf '\033[32m✓ %s\033[0m\n' "$*"; }
info(){ printf '\033[36m▸ %s\033[0m\n' "$*"; }

# ── Guards ────────────────────────────────────────────────────────────────────
[ "$(uname -s)" = "Darwin" ] || err "iOS projects can only be generated on macOS."
command -v node >/dev/null 2>&1 || err "node not found on PATH."
xcodebuild -version >/dev/null 2>&1 || err \
  "Full Xcode required. Install Xcode, then: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
command -v pod >/dev/null 2>&1 || err \
  "CocoaPods not found. Install:  brew install cocoapods   (or)  sudo gem install cocoapods"

# Exact RN version pinned in package.json (e.g. "0.80.0"), independent of whether it is installed yet.
RN_VERSION="$(node -p "require('$ROOT/package.json').dependencies['react-native'].replace(/[^0-9.]/g,'')")"
[ -n "$RN_VERSION" ] || err "Could not read the react-native version from package.json."
info "Target React Native: $RN_VERSION"

# ── 1. node_modules — required by `pod install` (use_native_modules! autolinking) ──
# NOTE: this repo's deps currently need --legacy-peer-deps (a react-navigation peer conflict).
if [ ! -d "$ROOT/node_modules/react-native" ]; then
  info "Installing node modules (first run; --legacy-peer-deps)…"
  ( cd "$ROOT" && npm install --legacy-peer-deps --no-audit --no-fund )
fi

# ── 2. Scaffold a throwaway project named FedLearn at the pinned RN version ─────
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
info "Scaffolding RN $RN_VERSION template via @react-native-community/cli…"
npx --yes "@react-native-community/cli@latest" init "$APP" \
  --version "$RN_VERSION" \
  --directory "$STAGE/$APP" \
  --package-name "com.fedlearn.mobile" \
  --skip-install --skip-git-init --install-pods false
SRC_IOS="$STAGE/$APP/ios"
[ -d "$SRC_IOS/$APP.xcodeproj" ] || err "Template scaffold did not produce $SRC_IOS/$APP.xcodeproj."

# ── 3. Install the generated Xcode project into ios/ (keep OUR sources + Podfile) ──
rm -rf "$IOS_DIR/$APP.xcodeproj"
cp -R "$SRC_IOS/$APP.xcodeproj" "$IOS_DIR/$APP.xcodeproj"
ok "Wrote $IOS_DIR/$APP.xcodeproj"

# ── 4. Add template support files we LACK (LaunchScreen, PrivacyInfo, main.*, .xcode.env)
#       without clobbering our custom AppDelegate.swift / Info.plist / Images.xcassets / etc. ──
mkdir -p "$IOS_DIR/$APP"
cp -Rn "$SRC_IOS/$APP/." "$IOS_DIR/$APP/" 2>/dev/null || true   # -n: ours win on conflict
[ -f "$SRC_IOS/.xcode.env" ] && cp -n "$SRC_IOS/.xcode.env" "$IOS_DIR/.xcode.env" 2>/dev/null || true

# ── 5. Wire the app-target native glue into the pbxproj (DeviceState.swift + bridging header +
#       header search paths). The C++ core/bridge/provider come in via FedLearnCore.podspec. ──
info "Wiring app-target native glue (ios/wire_native.rb)…"
gem list -i xcodeproj >/dev/null 2>&1 || gem install xcodeproj
ruby "$IOS_DIR/wire_native.rb"

# ── 6. Pods -> produces FedLearn.xcworkspace (uses OUR ios/Podfile) ────────────
# Native FL core is compiled in only when FEDLEARN_NATIVE_IOS=1 AND the libtorch/gRPC xcframework
# env vars (see FedLearnCore.podspec) are set; otherwise this builds the JS shell only.
info "Running pod install (RCT_NEW_ARCH_ENABLED=1)…"
( cd "$IOS_DIR" && RCT_NEW_ARCH_ENABLED=1 pod install )
ok "Wrote $IOS_DIR/$APP.xcworkspace"

cat <<EOF

$(ok "iOS project generated + native FL core wired.")
  Open:        open "$IOS_DIR/$APP.xcworkspace"
  Run (Metro): (cd "$ROOT" && npm run ios)            # JS shell (native core off)

The native C++ FL core is wired via FedLearnCore.podspec (compiles shared/ + bridge/ + gRPC) and the
FedLearnFactoryDelegate TurboModule hook; DeviceState.swift + the bridging header are in the target
(ios/wire_native.rb). To BUILD WITH the native core (VERIFY-BEFORE-BUILD):
  1. Build the iOS slices of the deps and the proto stubs:
       scripts/build_libtorch_arm64.sh   scripts/build_grpc_arm64.sh   (cd proto && buf generate)
  2. Point the podspec at them and re-install:
       export FEDLEARN_NATIVE_IOS=1
       export FEDLEARN_LIBTORCH_XCFRAMEWORK=/abs/libtorch.xcframework
       export FEDLEARN_GRPC_XCFRAMEWORKS=/abs/grpc.xcframework:/abs/absl.xcframework:...
       export FEDLEARN_PROTO_GEN_DIR=/abs/proto/gen/cpp
       (cd "$IOS_DIR" && RCT_NEW_ARCH_ENABLED=1 pod install)
  3. Set Development Team / signing (Signing & Capabilities) before a device build.
EOF
