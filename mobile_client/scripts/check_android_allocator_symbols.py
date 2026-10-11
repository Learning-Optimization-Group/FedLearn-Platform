#!/usr/bin/env python3
"""Reject a native library whose BoringSSL allocator can bind to Folly's data symbol."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library')
    parser.add_argument('--readelf', default='llvm-readelf')
    args = parser.parse_args()
    symbols = subprocess.check_output([args.readelf, '--dyn-syms', '--wide', args.library], text=True)
    isolated = [line for line in symbols.splitlines()
                if line.split() and line.split()[-1] == 'fedlearn_boringssl_sdallocx']
    if not isolated:
        raise SystemExit('Missing isolated BoringSSL allocator reference')
    if any('WEAK' not in line or 'UND' not in line for line in isolated):
        raise SystemExit('BoringSSL allocator hook must remain unresolved and weak')
    print('BoringSSL allocator reference is isolated from Folly')


if __name__ == '__main__':
    main()
