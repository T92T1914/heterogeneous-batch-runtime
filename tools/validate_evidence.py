"""Check retained numerical benchmark rows and their published byte identities."""
import csv
import hashlib
import importlib.util
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

# The SYCL and CUDA comparison: two runs under docs/sycl-cuda-protocol.json. The second run's
# receipt is published. The first run's receipt recorded a local path and stays private, so its
# CSV is checked against the hash and source that receipt holds. The results page must carry,
# under each run's heading, the tables report_sycl_cuda.py makes from that run's CSV, the
# recorded hashes and revisions, and the headline range and pair count that follow from both.
spec = importlib.util.spec_from_file_location('report_sycl_cuda', Path(__file__).resolve().parent / 'report_sycl_cuda.py')
report_sycl_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_sycl_cuda)
FIRST_SYCL_RUN = {'csv_sha256': '4d5ee831b4a92c18c83c8883f1e22eac4206a7a35ba176267dde3c533c475d79',
                  'source': '1c31bb781251e6561bb2e0a63c84b1b9ae357ff2'}
sycl_receipt = json.loads((root / 'sycl-cuda-execution.json').read_text(encoding='utf-8'))
sycl_bytes = (root / 'sycl-cuda.csv').read_bytes()
first_bytes = (root / 'sycl-cuda-attempt-1.csv').read_bytes()
if hashlib.sha256(sycl_bytes).hexdigest() != sycl_receipt['csv_sha256']:
    raise ValueError('Retained evidence changed: sycl-cuda.csv')
if hashlib.sha256(first_bytes).hexdigest() != FIRST_SYCL_RUN['csv_sha256']:
    raise ValueError('Retained evidence changed: sycl-cuda-attempt-1.csv')
if hashlib.sha256((root.parent / 'sycl-cuda-protocol.json').read_bytes()).hexdigest() != sycl_receipt['protocol_sha256']:
    raise ValueError('The SYCL and CUDA protocol changed after its runs')
if sycl_receipt['status'] != 'passed' or sycl_receipt['exit_code'] != 0:
    raise ValueError('The SYCL and CUDA run did not pass')
sycl_rows = list(csv.DictReader(sycl_bytes.decode('utf-8-sig').splitlines()))
first_rows = list(csv.DictReader(first_bytes.decode('utf-8-sig').splitlines()))
results_page = (root.parent / 'sycl-cuda-results.md').read_text(encoding='utf-8')
sections = {}
for part in ('\n' + results_page).split('\n## ')[1:]:
    heading, _, body = part.partition('\n')
    sections[heading] = body
for heading, tables in (('Second run', report_sycl_cuda.report(sycl_rows, sycl_receipt)),
                        ('First run', report_sycl_cuda.report(first_rows))):
    if tables not in sections.get(heading, ''):
        raise ValueError(f'The SYCL and CUDA results page differs from its samples under {heading}')
for value in (FIRST_SYCL_RUN['csv_sha256'], FIRST_SYCL_RUN['source'], sycl_receipt['csv_sha256'], sycl_receipt['source'],
              sycl_receipt['binary_sha256'], sycl_receipt['protocol_sha256'], sycl_receipt['protocol_commit']):
    if value not in results_page:
        raise ValueError(f'The SYCL and CUDA results page does not cite {value}')
gaps, slower = [], []
for rows in (first_rows, sycl_rows):
    runs = report_sycl_cuda.validate(rows)
    stats = report_sycl_cuda.summarize(runs)
    for side in report_sycl_cuda.SIDES:
        for workload in report_sycl_cuda.WORKLOADS:
            cuda, sycl = stats[(side, workload, 'cuda')]['host_ms'], stats[(side, workload, 'sycl')]['host_ms']
            gaps.append(100 * (1 - sycl['median'] / cuda['median']))
            pairs = zip(runs[(side, workload, 'cuda')][1:], runs[(side, workload, 'sycl')][1:])
            slower.append(sum(float(s['host_ms']) > float(c['host_ms']) for c, s in pairs))
headline = (f'The gap was between {min(gaps):.1f} and {max(gaps):.1f} percent of the CUDA median.',
            f'between {min(slower)} and {max(slower)} of the {report_sycl_cuda.PROTOCOL["warm_samples"]} back to back pairs had SYCL slower')
if any(text not in results_page for text in headline):
    raise ValueError(f'The SYCL and CUDA results page headline differs from its samples: {headline}')
print(f'{len(sycl_rows)} and {len(first_rows)} SYCL and CUDA rows, the receipt, the unchanged protocol, the results tables and the page citations passed')

# The pinned input run under docs/pinned-protocol.json as amended on 2026-11-24. Its receipt is
# published. The results page must carry under its Results heading the tables report_pinned.py
# makes from the samples and the receipt, the recorded hashes and revisions, and the headline
# numbers that follow from the samples.
spec = importlib.util.spec_from_file_location('report_pinned', Path(__file__).resolve().parent / 'report_pinned.py')
report_pinned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_pinned)
pinned_receipt = json.loads((root / 'pinned-execution.json').read_text(encoding='utf-8'))
pinned_bytes = (root / 'pinned.csv').read_bytes()
if hashlib.sha256(pinned_bytes).hexdigest() != pinned_receipt['csv_sha256']:
    raise ValueError('Retained evidence changed: pinned.csv')
if hashlib.sha256((root.parent / 'pinned-protocol.json').read_bytes()).hexdigest() != pinned_receipt['protocol_sha256']:
    raise ValueError('The pinned protocol changed after its run')
if pinned_receipt['status'] != 'passed' or pinned_receipt['exit_code'] != 0:
    raise ValueError('The pinned run did not pass')
pinned_rows = list(csv.DictReader(pinned_bytes.decode('utf-8-sig').splitlines()))
pinned_page = (root.parent / 'pinned-results.md').read_text(encoding='utf-8')
pinned_sections = {}
for part in ('\n' + pinned_page).split('\n## ')[1:]:
    heading, _, body = part.partition('\n')
    pinned_sections[heading] = body
if report_pinned.report(pinned_rows, pinned_receipt) not in pinned_sections.get('Results', ''):
    raise ValueError('The pinned results page differs from its samples')
for value in (pinned_receipt['csv_sha256'], pinned_receipt['source'], pinned_receipt['binary_sha256'],
              pinned_receipt['protocol_sha256'], pinned_receipt['protocol_commit']):
    if value not in pinned_page:
        raise ValueError(f'The pinned results page does not cite {value}')
pinned_runs, _ = report_pinned.validate(pinned_rows)
pinned_pairs = report_pinned.paired(pinned_runs)
if any(not p['all']['p25'] <= 1 <= p['all']['p75'] for p in pinned_pairs.values()):
    raise ValueError('The pinned results page says every middle half of the paired ratios includes 1')
ratios = [p['all']['median'] for p in pinned_pairs.values()]
pinned_slower = [sum(float(b['host_ms']) > float(a['host_ms'])
                     for a, b in zip(pinned_runs[(side, workload, 'pageable')][1:], pinned_runs[(side, workload, 'pinned')][1:]))
                 for side, workload in pinned_pairs]
warm = report_pinned.PROTOCOL['warm_samples']
headline = (f'the median of the {warm} paired ratios was between {min(ratios):.3f} and {max(ratios):.3f}',
            f'between {min(pinned_slower)} and {max(pinned_slower)} of the {warm} back to back pairs had the pinned call slower')
if any(text not in pinned_page for text in headline):
    raise ValueError(f'The pinned results page headline differs from its samples: {headline}')
print(f'{len(pinned_rows)} pinned rows, the receipt, the unchanged protocol, the results tables, the page citations and the headline passed')

# The same plan again at PCIe generation 4 under docs/pinned-gen4-protocol.json. Its receipt holds
# the link readings, which report_pinned.py checks against the protocol's link before it makes the
# tables. The results page must also carry, under its comparison heading, the table that sets each
# condition of the generation 3 run beside this one.
gen4_receipt = json.loads((root / 'pinned-gen4-execution.json').read_text(encoding='utf-8'))
gen4_bytes = (root / 'pinned-gen4.csv').read_bytes()
if hashlib.sha256(gen4_bytes).hexdigest() != gen4_receipt['csv_sha256']:
    raise ValueError('Retained evidence changed: pinned-gen4.csv')
if hashlib.sha256((root.parent / 'pinned-gen4-protocol.json').read_bytes()).hexdigest() != gen4_receipt['protocol_sha256']:
    raise ValueError('The pinned generation 4 protocol changed after its run')
if gen4_receipt['status'] != 'passed' or gen4_receipt['exit_code'] != 0:
    raise ValueError('The pinned generation 4 run did not pass')
gen4_protocol = json.loads((root.parent / 'pinned-gen4-protocol.json').read_text(encoding='utf-8'))
if gen4_receipt['pcie_link'] != {'generation': gen4_protocol['pcie_link']['generation'],
                                 'width': gen4_protocol['pcie_link']['width']}:
    raise ValueError('The pinned generation 4 receipt names another link than its protocol')
gen4_rows = list(csv.DictReader(gen4_bytes.decode('utf-8-sig').splitlines()))
gen4_page = (root.parent / 'pinned-gen4-results.md').read_text(encoding='utf-8')
gen4_sections = {}
for part in ('\n' + gen4_page).split('\n## ')[1:]:
    heading, _, body = part.partition('\n')
    gen4_sections[heading] = body
if report_pinned.report(gen4_rows, gen4_receipt) not in gen4_sections.get('Results', ''):
    raise ValueError('The pinned generation 4 results page differs from its samples')
if (report_pinned.compare(pinned_rows, gen4_rows, 'generation 3', 'generation 4')
        not in gen4_sections.get('Beside the generation 3 run', '')):
    raise ValueError('The pinned generation 4 comparison differs from the samples of both runs')
for value in (gen4_receipt['csv_sha256'], gen4_receipt['source'], gen4_receipt['binary_sha256'],
              gen4_receipt['protocol_sha256'], gen4_receipt['protocol_commit']):
    if value not in gen4_page:
        raise ValueError(f'The pinned generation 4 results page does not cite {value}')
gen4_runs, _ = report_pinned.validate(gen4_rows)
gen4_pairs = report_pinned.paired(gen4_runs)
if any(not p['all']['p25'] <= 1 <= p['all']['p75'] for p in gen4_pairs.values()):
    raise ValueError('The pinned generation 4 page says every middle half of the paired ratios includes 1')
gen4_link = gen4_receipt['pcie_link_readings']
if any((r['gen_current'], r['width_current'], r['gen_max'], r['width_max']) != (4, 16, 4, 16) for r in gen4_link):
    raise ValueError('The pinned generation 4 page says every link reading was generation 4 with 16 lanes')
gen4_ratios = [p['all']['median'] for p in gen4_pairs.values()]
gen4_slower = [sum(float(b['host_ms']) > float(a['host_ms'])
                   for a, b in zip(gen4_runs[(side, workload, 'pageable')][1:], gen4_runs[(side, workload, 'pinned')][1:]))
               for side, workload in gen4_pairs]
headline = (f'the median of the {warm} paired ratios was between {min(gen4_ratios):.3f} and {max(gen4_ratios):.3f}',
            f'between {min(gen4_slower)} and {max(gen4_slower)} of the {warm} back to back pairs had the pinned call slower',
            f'All {len(gen4_link)} readings gave generation 4 with 16 lanes')
if any(text not in gen4_page for text in headline):
    raise ValueError(f'The pinned generation 4 page headline differs from its samples: {headline}')
print(f'{len(gen4_rows)} pinned generation 4 rows, the receipt and its link readings, the unchanged protocol, '
      'the results and comparison tables, the page citations and the headline passed')
