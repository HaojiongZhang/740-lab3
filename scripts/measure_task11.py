"""Run in the lab container with PYTHONPATH=/workspace; preserve raw data."""
import argparse, csv, json, subprocess, time
from pathlib import Path
from profiler import Profiler, Task, Backend

ap = argparse.ArgumentParser()
ap.add_argument('label')
ap.add_argument('--probe', action='store_true')
ap.add_argument('--impls', default='staged')
ap.add_argument('--passes', default='1')
a = ap.parse_args()
out = Path('results/task11'); out.mkdir(parents=True, exist_ok=True)
if a.probe:
    for rep in range(1, 4):
        r = subprocess.run(['adb', 'shell', 'cd /data/local/tmp/profiler-kernels/dsp && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./conv_vtcm_probe'], capture_output=True, text=True, timeout=180)
        (out / f'probe_{rep}.txt').write_text(r.stdout + r.stderr)
        if r.returncode: raise RuntimeError(r.stderr)
        print(f'Probe {rep} saved', flush=True)
else:
    p = Profiler()
    with (out / f'{a.label}.csv').open('w', newline='') as f, (out / f'{a.label}.jsonl').open('w') as log:
        w = csv.DictWriter(f, fieldnames=['label','impl','p','rep','total_us','copy_us','compute_us','checksum'])
        w.writeheader()
        for passes in map(int, a.passes.split(',')):
            for rep in range(1,4):
                for impl in a.impls.split(','):
                    r = p.run(Task('stage',Backend.DSP,{'impl':impl,'p':passes,'mb':32}))
                    log.write(json.dumps(dict(label=a.label,impl=impl,p=passes,rep=rep,raw=r.raw,fields=r.fields,error=r.error))+'\n'); log.flush()
                    if not r.ok: raise RuntimeError(r.error or r.raw)
                    w.writerow(dict(label=a.label,impl=impl,p=passes,rep=rep,total_us=r.compute_us,**{k:r.fields[k] for k in ('copy_us','compute_us','checksum')})); f.flush()
                time.sleep(0.3)
            print(f'{a.label}: P={passes} saved',flush=True)
            time.sleep(2)
