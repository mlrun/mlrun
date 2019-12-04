#!/usr/bin/env python
"""Print last http test log"""

from pathlib import Path


def mtime(path: Path):
    return path.stat().st_mtime


if __name__ == '__main__':
    test_dir = sorted(Path('/tmp').glob('mlrun-test*'), key=mtime)[-1]
    log_file = test_dir / 'httpd.log'
    with log_file.open() as fp:
        print(fp.read())
    print(f'\n\n{test_dir}')
