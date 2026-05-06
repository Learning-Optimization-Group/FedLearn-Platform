// =============================================================================
// FedLearn Desktop — Hardware Probe
// =============================================================================
// One-shot detection of the local execution environment so the renderer can
// pre-select the right profile. We only surface information the user can
// actually act on — if a CUDA GPU is visible, default to 'discrete'; if we're
// on Apple Silicon, default to 'mps'; otherwise default to 'cpu'.
//
// The detection is best-effort and intentionally silent on failure: the
// user can always override via the HardwareSelector cards.
// =============================================================================

import { app } from 'electron';
import { execFile } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import log from 'electron-log';

import type { HardwareProfile } from './docker.service';

export interface HardwareDetection {
  platform: NodeJS.Platform;
  arch: string;
  recommendedProfile: HardwareProfile;
  nativeBundleAvailable: boolean;
  cudaAvailable: boolean;
  cudaInfo?: string;
}

const NATIVE_BUNDLE_DIR = 'fedlearn-client';

function probeNvidiaSmi(): Promise<{ available: boolean; info?: string }> {
  return new Promise((resolve) => {
    // 2-second timeout — if nvidia-smi hangs (rare but happens on degraded
    // driver installs) we don't want to block the UI.
    const timeout = setTimeout(() => resolve({ available: false }), 2000);

    execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], (err, stdout) => {
      clearTimeout(timeout);
      if (err) {
        resolve({ available: false });
        return;
      }
      const info = stdout.trim().split('\n')[0] || 'NVIDIA GPU';
      resolve({ available: true, info });
    });
  });
}

function nativeBundleExists(): boolean {
  if (!app.isPackaged) return true; // dev mode uses python3 + scripts dir

  const binaryName = process.platform === 'win32' ? 'fedlearn-client.exe' : 'fedlearn-client';
  const binary = path.join(process.resourcesPath, NATIVE_BUNDLE_DIR, binaryName);
  return fs.existsSync(binary);
}

export async function detectHardware(): Promise<HardwareDetection> {
  const platform = process.platform;
  const arch = process.arch;
  const bundleAvailable = nativeBundleExists();
  const { available: cudaAvailable, info: cudaInfo } = await probeNvidiaSmi();

  let recommendedProfile: HardwareProfile;

  if (platform === 'darwin' && arch === 'arm64') {
    recommendedProfile = 'mps';
  } else if (cudaAvailable && platform !== 'linux') {
    // Non-Jetson CUDA. Linux-x64-CUDA isn't one of our shipped bundles yet,
    // so we fall through to cpu there and the user can flip to 'discrete'
    // manually if they want to go through Docker.
    recommendedProfile = 'discrete';
  } else {
    recommendedProfile = 'cpu';
  }

  const detection: HardwareDetection = {
    platform,
    arch,
    recommendedProfile,
    nativeBundleAvailable: bundleAvailable,
    cudaAvailable,
    cudaInfo,
  };

  log.info(`[HardwareProbe] ${JSON.stringify(detection)}`);
  return detection;
}
