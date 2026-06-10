# FedLearnCore.podspec — compiles the shared C++ federated-learning core (../shared) + gRPC layer +
# the CXX-TurboModule bridge (../bridge/common, ../bridge/ios) into the iOS app target, and links the
# cross-compiled libtorch + gRPC xcframeworks. This is how the native FL core is wired into iOS
# (the analogue of android/app/build.gradle's externalNativeBuild → bridge/android/jni/CMakeLists.txt).
#
# Included only when ENV['FEDLEARN_NATIVE_IOS'] is set (see Podfile) — exactly like Android's native
# build needs LIBTORCH_DIR/GENERATED_PROTO_DIR. Until the artifacts exist, the JS app still builds and
# `import FedLearnCore` is simply absent (AppDelegate falls back via `#if canImport`).
#
# VERIFY-BEFORE-BUILD: provide these absolute paths in the environment before `pod install`:
#   FEDLEARN_LIBTORCH_XCFRAMEWORK   = /abs/libtorch.xcframework        (scripts/build_libtorch_arm64.sh, iOS slice)
#   FEDLEARN_GRPC_XCFRAMEWORKS      = /abs/grpc.xcframework:/abs/...   (colon-separated; build_grpc_arm64.sh, iOS slice)
#   FEDLEARN_PROTO_GEN_DIR          = /abs/proto/gen/cpp              (`buf generate` from proto/)
require 'json'

pkg = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

libtorch = ENV['FEDLEARN_LIBTORCH_XCFRAMEWORK']
proto    = ENV['FEDLEARN_PROTO_GEN_DIR']
grpc_fw  = (ENV['FEDLEARN_GRPC_XCFRAMEWORKS'] || '').split(':').reject(&:empty?)

if libtorch.nil? || libtorch.empty? || proto.nil? || proto.empty?
  raise <<~MSG
    [FedLearnCore] native iOS build requires the cross-compiled artifacts. Set, then re-run `pod install`:
      export FEDLEARN_LIBTORCH_XCFRAMEWORK=/abs/path/libtorch.xcframework
      export FEDLEARN_GRPC_XCFRAMEWORKS=/abs/grpc.xcframework:/abs/absl.xcframework:...
      export FEDLEARN_PROTO_GEN_DIR=/abs/path/proto/gen/cpp
    Build them with scripts/build_libtorch_arm64.sh, scripts/build_grpc_arm64.sh (iOS slices) and
    `cd proto && buf generate`. To build the JS app WITHOUT the native core, unset FEDLEARN_NATIVE_IOS.
  MSG
end

Pod::Spec.new do |s|
  s.name             = 'FedLearnCore'
  s.version          = pkg['version'] || '0.0.0'
  s.summary          = 'Native C++ federated-learning core (DeComFL/FedAvg) + CXX TurboModule for iOS.'
  s.homepage         = 'https://github.com/anurag2796/FedLearn-Platform'
  s.license          = { :type => 'Proprietary' }
  s.author           = 'FedLearn'
  s.platforms        = { :ios => '15.1' }
  s.source           = { :path => '.' }
  s.requires_arc     = true

  # Paths are relative to this podspec (mobile_client/ios/).
  s.source_files = [
    '../shared/src/**/*.cpp',
    '../shared/include/**/*.h',
    '../bridge/common/**/*.{h,cpp}',
    '../bridge/ios/**/*.{h,mm}',
    "#{proto}/fedlearn/v2/**/*.{cc,h}",
  ]
  # Swift-visible / bridging-header-visible headers.
  s.public_header_files = [
    '../bridge/ios/FedLearnFactoryDelegate.h',
    '../bridge/common/DeviceState.h',
  ]

  s.vendored_frameworks = [libtorch] + grpc_fw

  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY'           => 'libc++',
    'DEFINES_MODULE'              => 'YES',   # so the Swift app can `import FedLearnCore`
    'HEADER_SEARCH_PATHS'         => [
      '"$(PODS_TARGET_SRCROOT)/../shared/include"',
      '"$(PODS_TARGET_SRCROOT)/../bridge/common"',
      %Q("#{proto}"),
    ].join(' '),
    # arm64 device + arm64 simulator only (matches the Android arm64-v8a-only ship target).
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'x86_64',
  }

  # RN runtime the bridge + factory delegate compile against (version-specific; VERIFY-BEFORE-BUILD).
  s.dependency 'React-Core'
  s.dependency 'ReactCommon'
  s.dependency 'React-RCTAppDelegate'   # RCTDefaultReactNativeFactoryDelegate base class
end
