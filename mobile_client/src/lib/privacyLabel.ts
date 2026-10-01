// The privacy label — the static, truthful data-flow disclosure ProjectDetail renders as the
// single interstitial before a join. Copy rules:
//   · client-side facts only, no marketing ("your data helps the world" is banned);
//   · nothing the code can't back: the raw dataset stays on-device (the native core trains locally and only
//     the example count is sent with an update), the wire carries learning updates (safetensors weights, or
//     DeComFL's gradient scalars and seeds), and this app ships no analytics/ads SDKs;
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
      'Your raw training data — the dataset you import, or the demo sample a test run provides. It is read '
        + 'locally for training and never uploaded.',
      'Which examples you have and their labels: the server learns only how many examples trained each update.',
    ],
  },
  {
    key: 'leaves',
    heading: 'Leaves your phone',
    points: [
      'Learning updates: the updated model weights as sha256-integrity-verified safetensors, or, on a '
        + 'low-bandwidth run, a few gradient numbers and the seeds they belong to.',
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
