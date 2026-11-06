const INVALID_ORIGIN = 'Enter an HTTP or HTTPS server origin without credentials, a path, query or fragment.';

function ipv6Host(raw: string): string {
  if (!/^[0-9a-f:]+$/i.test(raw)) throw new Error(INVALID_ORIGIN);
  const halves = raw.split('::');
  if (halves.length > 2) throw new Error(INVALID_ORIGIN);
  const left = halves[0] ? halves[0].split(':') : [];
  const right = halves.length === 2 && halves[1] ? halves[1].split(':') : [];
  const valid = (part: string) => /^[0-9a-f]{1,4}$/i.test(part);
  if (!left.every(valid) || !right.every(valid)
      || (halves.length === 1 ? left.length !== 8 : left.length + right.length >= 8)) {
    throw new Error(INVALID_ORIGIN);
  }
  const parts = [...left, ...Array<string>(8 - left.length - right.length).fill('0'), ...right]
    .map((part) => parseInt(part, 16).toString(16));
  let bestStart = -1;
  let bestLength = 1;
  for (let start = 0; start < parts.length;) {
    if (parts[start] !== '0') { start++; continue; }
    let end = start;
    while (end < parts.length && parts[end] === '0') end++;
    if (end - start > bestLength) { bestStart = start; bestLength = end - start; }
    start = end;
  }
  const host = bestStart < 0 ? parts.join(':')
    : parts.slice(0, bestStart).join(':') + '::' + parts.slice(bestStart + bestLength).join(':');
  return '[' + host + ']';
}

/** Canonical backend authority using the supported ASCII-host/hex-IPv6 grammar. */
export function normalizeBackendOrigin(raw: string): string {
  const value = (raw ?? '').trim().replace(/\/+$/, '');
  const match = /^(https?):\/\/(\[[0-9a-f:]+\]|[a-z0-9.-]+)(?::([0-9]{1,5}))?$/i.exec(value);
  if (!match) throw new Error(INVALID_ORIGIN);
  const scheme = match[1]!.toLowerCase();
  let host = match[2]!.toLowerCase();
  if (host.startsWith('[')) {
    host = ipv6Host(host.slice(1, -1));
  } else {
    const labels = host.split('.');
    if (host.length > 253 || labels.some((label) => !/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/i.test(label))) {
      throw new Error(INVALID_ORIGIN);
    }
    if (/^[0-9.]+$/.test(host)) {
      if (labels.length !== 4 || labels.some((label) => !/^(0|[1-9][0-9]{0,2})$/.test(label) || Number(label) > 255)) {
        throw new Error(INVALID_ORIGIN);
      }
    } else if (/^(?:0x[0-9a-f]+|[0-9]+)$/i.test(labels[labels.length - 1]!)) {
      throw new Error(INVALID_ORIGIN);
    }
  }
  const port = match[3] ? Number(match[3]) : scheme === 'https' ? 443 : 80;
  if (port < 1 || port > 65535) throw new Error(INVALID_ORIGIN);
  return scheme + '://' + host + ((scheme === 'https' && port === 443) || (scheme === 'http' && port === 80) ? '' : ':' + port);
}

/** Resolve only the destination authority; ambiguous targets cannot receive a credential. */
export function requestOrigin(base: string | undefined, url?: string): string | null {
  try {
    if (url && /[\s\\]/.test(url)) return null;
    if (url && /^https?:\/\//i.test(url)) {
      const authority = /^(https?:\/\/[^/?#]+)/i.exec(url);
      return authority?.[1] ? normalizeBackendOrigin(authority[1]) : null;
    }
    if (!base) return null;
    const origin = normalizeBackendOrigin(base);
    if (url?.startsWith('//')) return requestOrigin(undefined, (origin.startsWith('https:') ? 'https:' : 'http:') + url);
    if (!url || url.startsWith('/')) return origin;
    return null;
  } catch {
    return null;
  }
}
