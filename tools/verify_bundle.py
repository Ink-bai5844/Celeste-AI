"""Verify every packaged file against the handoff checksum manifest."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'handoff' / 'MANIFEST.sha256'
    if not manifest.is_file():
        raise SystemExit('Run this tool from the extracted handoff package, which contains MANIFEST.sha256.')
    failures, count = [], 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            failures.append(name)
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        count += 1
        if digest.hexdigest() != expected:
            failures.append(name)
    print(f'Checked {count} files; mismatches: {len(failures)}')
    for name in failures:
        print(name)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
