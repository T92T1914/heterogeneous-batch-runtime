"""Check retained numerical benchmark rows and their published byte identities."""
import csv
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'docs' / 'evidence'
receipt = json.loads((root / 'execution.json').read_text(encoding='utf-8'))
for name, expected in receipt['artifact_sha256'].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Retained evidence changed: {name}')
for name, count in [('baseline-cpu.csv',162),('followup-cpu.csv',162),('baseline-cuda.csv',180)]:
    with (root/name).open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=count:
        raise ValueError(f'Incomplete conditions in {name}')
    seen=set()
    groups={}
    references={}
    for row in rows:
        key=(row['workload'],row['side'],row.get('distribution',''),row['backend'],row.get('threads',''))
        identity=(*key,row['iteration'])
        if identity in seen:
            raise ValueError(f'Duplicate condition {identity}')
        seen.add(identity)
        groups.setdefault(key,set()).add(int(row['iteration']))
        for field in ('elapsed_ms','upload_ms','kernel_ms','download_ms','host_total_ms'):
            if field in row and (not math.isfinite(float(row[field])) or float(row[field])<0):
                raise ValueError(f'Invalid timing {name}: {field}')
        checksum=float(row['checksum'])
        if not math.isfinite(checksum):
            raise ValueError('Nonfinite output checksum')
        source_key=(row['workload'],row['side'],row.get('distribution',''))
        reference=references.setdefault(source_key,checksum)
        tolerance=1e-8 if row['workload']=='masked_reduce' else 1e-12
        if abs(checksum-reference)>tolerance:
            raise ValueError(f'Cross-backend output checksum differs: {source_key}')
    if any(values!=set(range(6)) for values in groups.values()):
        raise ValueError('Missing first-invocation or warm sample')
for check in receipt['local_verification']:
    if check['status']!='passed' or any(c['status']!='passed' or c['exit_code']!=0 for c in check['checks']):
        raise ValueError('Claimed local validation did not pass')
print('504 retained timing rows, artifact hashes and local verification summaries passed')
