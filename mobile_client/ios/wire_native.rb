#!/usr/bin/env ruby
# wire_native.rb — adds the FedLearn app-target glue that lives OUTSIDE CocoaPods to
# FedLearn.xcodeproj (the C++ core + bridge are handled by FedLearnCore.podspec; these are the
# app-target Swift/bridging bits the pod can't add):
#   1. FedLearn/DeviceState.swift  -> the app target's compile sources
#   2. SWIFT_OBJC_BRIDGING_HEADER  -> FedLearn/FedLearn-Bridging-Header.h
#   3. HEADER_SEARCH_PATHS         -> ../bridge/common, ../shared/include (so the bridging header
#                                     resolves DeviceState.h even before the native pod is installed)
#
# Idempotent — generate_xcodeproj.sh calls this after creating the project; safe to re-run.
# Requires the `xcodeproj` gem (`gem install xcodeproj`), the same library CocoaPods uses.
require 'xcodeproj'

ios_dir   = File.dirname(File.expand_path(__FILE__))
proj_path = File.join(ios_dir, 'FedLearn.xcodeproj')
abort("✗ #{proj_path} not found — run ./generate_xcodeproj.sh first") unless File.exist?(proj_path)

proj   = Xcodeproj::Project.open(proj_path)
target = proj.targets.find { |t| t.name == 'FedLearn' } or abort('✗ FedLearn target not found')
group  = proj.main_group.find_subpath('FedLearn', true)

# 1. DeviceState.swift -> app target sources (idempotent)
already = target.source_build_phase.files_references.any? { |r| r.path&.end_with?('DeviceState.swift') }
if already
  puts '= DeviceState.swift already in target'
else
  ref = group.new_reference('DeviceState.swift')
  target.add_file_references([ref])
  puts '+ added FedLearn/DeviceState.swift to the FedLearn target'
end

# 2 + 3. bridging header + header search paths on every build configuration
extra_paths = ['$(SRCROOT)/../bridge/common', '$(SRCROOT)/../shared/include']
target.build_configurations.each do |c|
  c.build_settings['SWIFT_OBJC_BRIDGING_HEADER'] = 'FedLearn/FedLearn-Bridging-Header.h'
  hsp = c.build_settings['HEADER_SEARCH_PATHS'] || ['$(inherited)']
  hsp = [hsp] unless hsp.is_a?(Array)
  extra_paths.each { |p| hsp << p unless hsp.include?(p) }
  c.build_settings['HEADER_SEARCH_PATHS'] = hsp
end
puts '+ set SWIFT_OBJC_BRIDGING_HEADER + HEADER_SEARCH_PATHS on all configs'

proj.save
puts "✓ wired app-target glue into #{File.basename(proj_path)}"
