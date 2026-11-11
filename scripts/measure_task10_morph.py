"""Run in /workspace: python scripts/measure_task10_morph.py.

Serial paired measurements; incremental CSV and raw logs, six repetitions.
"""
import csv
import json
from pathlib import Path
import time
from profiler import Profiler, Task, Backend

out = Path('results/task10_morph')
out.mkdir(parents=True, exist_ok=True)
p = Profiler()
with (out / 'measurements.csv').open('w', newline='') as f, (out / 'raw.jsonl').open('w') as log:
    writer = csv.DictWriter(f, fieldnames=['op', 'k', 'impl', 'rep', 'compute_us', 'residual', 'fingerprint'])
    writer.writeheader()
    for op in ('erode', 'dilate'):
        for k in (3, 5, 7):
            for rep in range(1, 7):
                for impl in ('student', 'qhl'):
                    r = p.run(Task('morph', Backend.DSP, {'op': op, 'k': k, 'impl': impl, 'input': 'dog.pgm'}))
                    log.write(json.dumps({'op': op, 'k': k, 'impl': impl, 'rep': rep,
                                          'raw': r.raw, 'fields': r.fields, 'error': r.error}) + '\n')
                    log.flush()
                    if not r.ok or r.compute_us is None:
                        raise RuntimeError(r.error or r.raw)
                    assert int(r.fields['W']) == 1024 and int(r.fields['H']) == 768, r.raw
                    writer.writerow(dict(op=op, k=k, impl=impl, rep=rep, compute_us=r.compute_us,
                                         residual=r.fields['residual'], fingerprint=r.fingerprint))
                    f.flush()
                time.sleep(0.25)
            print(f'Completed {op} K={k}, six repetitions per implementation', flush=True)
            time.sleep(2)
