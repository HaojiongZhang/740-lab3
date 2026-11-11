"""Independent FP16 SAXPY oracle and repeat/fingerprint checks (NumPy)."""
import json
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parents[1]/'results/task13'
def read(name):return [json.loads(s) for s in (out/name).read_text().splitlines()]
rows=read('search.jsonl')
def golden(n):
    z=np.arange(1,2*n+1,dtype=np.uint64)*np.uint64(0x9e3779b97f4a7c15)+np.uint64(42)
    z=(z^(z>>30))*np.uint64(0xbf58476d1ce4e5b9)
    z=(z^(z>>27))*np.uint64(0x94d049bb133111eb);z=z^(z>>31)
    u=(z>>11).astype(np.float32)*np.float32(1/(1<<53))
    inp=(np.float32(2)*u-np.float32(1)).astype(np.float16)
    y=(np.float32(2)*inp[:n].astype(np.float32)+inp[n:].astype(np.float32)).astype(np.float16).astype(np.float64)
    return float(y.sum()),float(np.sqrt((y*y).sum()))
checks=[]
for r in rows:
    if not r['ok']: continue  # Preserve failed supplemental attempts; never use their timings.
    n=int(r['params']['size']);s,f=golden(n)
    ds=abs(s-float(r['fields']['fingerprint_sum']));df=abs(f-float(r['fields']['fingerprint_frob']))
    assert ds<2e-5 and df<2e-5,(n,ds,df)
    checks.append(dict(size=n,label=r['label'],sum_abs_error=ds,frob_abs_error=df))
main=read('main.jsonl');startup=read('startup.jsonl')
assert len(main)==27 and len(startup)==24
for label in ('int8','finalize512','cpu1','cpu8','saxpy_cpu','gpu512'):
    selected=[r for r in main if r['label']==label]
    for n in {r['params']['size'] for r in selected}:
        fps={r['fingerprint'] for r in selected if r['params']['size']==n}
        assert len(fps)==1,(label,n,fps)
for r in startup:
    assert r['ok'] and r['temp_peak_c']<58
# The GPU repeats inside one kernel and divides the accumulated result by R:
# FP32 rounding increases with R. Compare relatively, not as exact equality.
gpu_one=next(r['fingerprint'] for r in startup if r['label']=='startup_gpu' and r['params']['inner_reps']==1)
for r in startup:
    if r['label']=='startup_gpu':assert abs(r['fingerprint']/gpu_one-1)<.0003
    else:assert abs(r['fingerprint']/gpu_one-1)<.00001
np_fp=next(r['fingerprint'] for r in startup if r['label']=='startup_npu')
assert abs(np_fp/gpu_one-1)<.00001
report=dict(saxpy_reference_checks=checks,main_processes=len(main),startup_processes=len(startup),
    startup_peak_c=max(r['temp_peak_c'] for r in startup),gpu_npu_relative_fingerprint_difference=abs(np_fp/gpu_one-1),
    limitations=['CPU matmul uses different inputs from GPU/NPU; INT8 uses its own quantized inputs.',
    'SAXPY CPU 50000 iterations updates y in place; NPU reuses fixed inputs.',
    'Repeated GPU inner-loop fingerprints have expected accumulation rounding; tolerance 0.03%.',
    'INT8 is validated for harness success and stable fingerprints, not equivalence to FP16 inputs.'])
(out/'validation.json').write_text(json.dumps(report,indent=2))
print(f'PASS: {len(checks)} SAXPY outputs match independent FP16 oracle; repeats stable; startup fingerprints within documented precision tolerances.')
