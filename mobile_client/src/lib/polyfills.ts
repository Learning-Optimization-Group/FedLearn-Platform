// Guarded TextEncoder/TextDecoder polyfill. @stomp/stompjs (used by stompClient.ts for the live log
// and streaming-chat features) encodes/decodes STOMP frames with TextEncoder/TextDecoder, which the
// RN 0.80 JS runtime does not guarantee as globals. If Hermes already provides them this is a no-op;
// otherwise we install a correct, minimal UTF-8 implementation so the STOMP path never throws
// "TextEncoder is not defined" at connect time. Imported first, before anything else, in index.js.
/* eslint-disable @typescript-eslint/no-explicit-any */

const g = globalThis as any;

if (typeof g.TextEncoder === 'undefined') {
  g.TextEncoder = class {
    readonly encoding = 'utf-8';
    encode(str = ''): Uint8Array {
      const out: number[] = [];
      for (let i = 0; i < str.length; i++) {
        let c = str.charCodeAt(i);
        if (c < 0x80) {
          out.push(c);
        } else if (c < 0x800) {
          out.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
        } else if (c >= 0xd800 && c < 0xdc00) {
          // High surrogate → combine with the following low surrogate into a code point.
          const c2 = str.charCodeAt(++i);
          c = 0x10000 + ((c & 0x3ff) << 10) + (c2 & 0x3ff);
          out.push(
            0xf0 | (c >> 18),
            0x80 | ((c >> 12) & 0x3f),
            0x80 | ((c >> 6) & 0x3f),
            0x80 | (c & 0x3f),
          );
        } else {
          out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
        }
      }
      return new Uint8Array(out);
    }
  };
}

if (typeof g.TextDecoder === 'undefined') {
  g.TextDecoder = class {
    readonly encoding = 'utf-8';
    decode(input?: ArrayBuffer | ArrayBufferView): string {
      if (!input) return '';
      const bytes =
        input instanceof Uint8Array
          ? input
          : new Uint8Array(input instanceof ArrayBuffer ? input : (input as ArrayBufferView).buffer);
      let out = '';
      let i = 0;
      const next = (): number => bytes[i++] ?? 0; // trailing bytes → 0 (defensive; well-formed UTF-8)
      while (i < bytes.length) {
        const c = next();
        if (c < 0x80) {
          out += String.fromCharCode(c);
        } else if (c < 0xe0) {
          out += String.fromCharCode(((c & 0x1f) << 6) | (next() & 0x3f));
        } else if (c < 0xf0) {
          out += String.fromCharCode(((c & 0x0f) << 12) | ((next() & 0x3f) << 6) | (next() & 0x3f));
        } else {
          const cp =
            ((c & 0x07) << 18) | ((next() & 0x3f) << 12) | ((next() & 0x3f) << 6) | (next() & 0x3f);
          const u = cp - 0x10000;
          out += String.fromCharCode(0xd800 + (u >> 10), 0xdc00 + (u & 0x3ff));
        }
      }
      return out;
    }
  };
}

export {};
