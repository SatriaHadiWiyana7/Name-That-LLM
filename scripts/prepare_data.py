#!/usr/bin/env python3
"""Safely extract ONLY the named competition dataset files from user-owned ZIP."""
import argparse
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip', required=True, dest='archive')
parser.add_argument('--out', default='data/raw')
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with ZipFile(args.archive) as z:
    names = {Path(name).name: name for name in z.namelist() if not name.endswith('/')}
    for fname in ('train.csv','test.csv','sample_submission.parquet'):
        if fname not in names:
            raise SystemExit(f'Missing {fname} in ZIP')
        contents = z.read(names[fname])
        (out/fname).write_bytes(contents)
        print(f'{fname}: {len(contents):,} bytes -> {out/fname}')
print('Dataset remains local/private. Do not commit data/raw or publicly share.')
