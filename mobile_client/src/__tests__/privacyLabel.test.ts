// The privacy label is the single interstitial before a join — pin its three sections and the
// honesty rules of its copy.
import { PRIVACY_SECTIONS } from '../lib/privacyLabel';

describe('PRIVACY_SECTIONS', () => {
  it('has exactly the three sections, in disclosure order', () => {
    expect(PRIVACY_SECTIONS.map((s) => s.heading)).toEqual([
      'Stays on your phone',
      'Leaves your phone',
      'Never collected',
    ]);
    expect(PRIVACY_SECTIONS.map((s) => s.key)).toEqual(['stays', 'leaves', 'never']);
  });

  it('every section carries at least one non-empty point', () => {
    for (const section of PRIVACY_SECTIONS) {
      expect(section.points.length).toBeGreaterThan(0);
      for (const point of section.points) {
        expect(point.trim().length).toBeGreaterThan(0);
      }
    }
  });

  it('grounds the wire claims: raw data stays, updates leave as sha256-verified safetensors', () => {
    const stays = PRIVACY_SECTIONS.find((s) => s.key === 'stays');
    const leaves = PRIVACY_SECTIONS.find((s) => s.key === 'leaves');
    expect(stays?.points.join(' ')).toContain('raw training data');
    expect(stays?.points.join(' ')).toContain('partition');
    expect(leaves?.points.join(' ')).toContain('sha256');
    expect(leaves?.points.join(' ')).toContain('safetensors');
  });

  it('stays plain text — no emoji or glyph bullets (C5 §9)', () => {
    for (const section of PRIVACY_SECTIONS) {
      for (const text of [section.heading, ...section.points]) {
        // Ledger copy is ASCII prose plus typographic dashes/quotes; emoji live outside BMP ranges.
        expect(/[\u{1F000}-\u{1FFFF}\u{2600}-\u{27BF}]/u.test(text)).toBe(false);
      }
    }
  });
});
