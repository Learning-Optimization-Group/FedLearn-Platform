// The privacy label — the static, truthful data-flow disclosure ProjectDetail renders as the
// single interstitial before a join. Copy rules:
//   · client-side facts only, no marketing ("your data helps the world" is banned);
//   · nothing the code can't back: the raw dataset and partition stay on-device
//     (fl-runtime/native core trains locally), the wire carries model learning updates as
//     sha256-integrity-verified safetensors, and this app ships no analytics/ads SDKs;
//   · the live server endpoint for a joined run is appended by the screen (dynamic, not here).
// Pure data so tests can pin the three section headings without a renderer.

export type PrivacySectionKey = 'stays' | 'leaves' | 'never';

export interface PrivacySection {
  key: PrivacySectionKey;
  heading: string;
  points: readonly string[];
}

export const PRIVACY_SECTIONS: readonly PrivacySection[] = [
  {
    key: 'stays',
    heading: 'Stays on your phone',
    points: [
      'Your raw training data — it is read locally for training and never uploaded.',
      'Your dataset partition: the slice of data this device is assigned to train on.',
    ],
  },
  {
    key: 'leaves',
    heading: 'Leaves your phone',
    points: [
      'Model weight updates only, sent as sha256-integrity-verified safetensors.',
      'While joined, the app talks to the training server shown below.',
    ],
  },
  {
    key: 'never',
    heading: 'Never collected',
    points: [
      'Photos, messages, contacts, or location — the app never reads them.',
      'No analytics or advertising identifiers: this app contains no tracking SDKs.',
    ],
  },
] as const;
