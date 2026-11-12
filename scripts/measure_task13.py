"""Task 13 measurements. Run in /workspace with PYTHONPATH=/workspace."""
import argparse,atexit,json,time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from profiler import Profiler,Backend,matmul,saxpy,ThermalReader

ap=argparse.ArgumentParser();ap.add_argument('part',choices=['main','search','startup']);ap.add_argument('--cool-compiler',action='store_true');ap.add_argument('--compiler-cap',type=int,default=3033600);ap.add_argument('--repeat-endpoint',action='store_true');a=ap.parse_args()
out=Path('results/task13');out.mkdir(parents=True,exist_ok=True)
p=Profiler()
if a.cool_compiler:
    original_shell=p.device.shell
    node='/sys/devices/system/cpu/cpufreq/policy0/'
    old_min=original_shell('cat '+node+'scaling_min_freq')[0].strip()
    old_max=original_shell('cat '+node+'scaling_max_freq')[0].strip()
    def restore():
        original_shell("su 0 sh -c 'echo "+old_max+' > '+node+"scaling_max_freq; echo "+old_min+' > '+node+"scaling_min_freq'",timeout=10)
    atexit.register(restore)
    cap=str(a.compiler_cap)
    rc=original_shell("su 0 sh -c 'echo 384000 > "+node+'scaling_min_freq; echo '+cap+' > '+node+"scaling_max_freq'")[2]
    assert rc==0 and original_shell('cat '+node+'scaling_max_freq')[0].strip()==cap
    (out/'compiler_clock_state.json').write_text(json.dumps(dict(original_min=old_min,original_max=old_max,cap_khz=a.compiler_cap,affinity='0x3f')))
    def cool_shell(cmd,**kw):
        if './saxpy_qnn_fp16 ' in cmd:cmd=cmd.replace('./saxpy_qnn_fp16 ','taskset 3f ./saxpy_qnn_fp16 ')
        return original_shell(cmd,**kw)
    p.device.shell=cool_shell
thermals=ThermalReader(p.device)
path=out/f'{a.part}.jsonl'
previous=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
f=path.open('a')
def run(label,task,rep=1):
    before=thermals.read()
    if not before: raise RuntimeError('No temperature telemetry')
    while max(before.values())>38:
        print(f'Cooling: {max(before.values()):.1f} C; waiting for <=38 C',flush=True)
        time.sleep(10); before=thermals.read()
        if not before: raise RuntimeError('No temperature telemetry')
    peaks=[max(before.values())]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(p.run,task)
        while True:
            try:
                r=future.result(timeout=3);break
            except FutureTimeout:
                t=thermals.read()
                if not t or max(t.values())>=58:
                    with (out/'thermal_stops.jsonl').open('a') as stoplog:
                        stoplog.write(json.dumps(dict(label=label,params=task.params,temps=t))+'\n')
                    print('Thermal guard telemetry: '+str(t),flush=True)
                    # Stop only the benchmark currently launched by this runner.
                    binary='saxpy_qnn_fp16' if task.workload=='saxpy' else ('matmul_qnn_fp16' if task.backend==Backend.NPU else 'matmul_opencl_fp32_3x16u4_sg_half')
                    p.device.shell('for n in $(pidof '+binary+'); do kill -TERM "$n"; done',timeout=10)
                    raise RuntimeError('Thermal guard stopped the workload')
                peaks.append(max(t.values()))
    logical_us=r.compute_us_per_iter
    if task.backend==Backend.GPU and 'duration_ns' in r.fields:
        logical_us=float(r.fields['duration_ns'])/1000/int(r.fields.get('inner_reps',1))
    rec=dict(label=label,rep=rep,params=task.params,variant=task.resolved_variant,
        backend=task.backend.value,ok=r.ok,error=r.error,fields=r.fields,raw=r.raw,
        per_iter_us=logical_us,fingerprint=r.fingerprint,gflops=r.gflops,
        temp_before_c=max(before.values()),temp_peak_c=max(peaks+[r.temp_c or 0]),temp_after_c=r.temp_c,cool_compiler=a.cool_compiler,compiler_cap_khz=a.compiler_cap if a.cool_compiler else None)
    f.write(json.dumps(rec)+'\n');f.flush()
    if not r.ok or r.compute_us_per_iter is None:raise RuntimeError(r.error or r.raw)
    print(f'{label} rep={rep} params={task.params} time={logical_us:.2f} us finalize={r.fields.get("finalize_us","-")}',flush=True)
    time.sleep(3)
    return r

if a.part=='main':
    for n in (256,512,1024):
        for rep in range(1,4):run('int8',matmul(Backend.NPU,variant='int8',size=n),rep)
    for rep in range(1,6):run('finalize512',matmul(Backend.NPU,size=512),rep)
    for rep in range(1,4):
        run('cpu1',matmul(Backend.CPU,size=512,iterations=40,core='7',threads=1),rep)
        run('cpu8',matmul(Backend.CPU,size=512,iterations=150,core='all',threads=8),rep)
        run('saxpy_cpu',saxpy(Backend.CPU,size=65536,iterations=50000,core='7'),rep)
        run('gpu512',matmul(Backend.GPU,variant='best',size=512),rep)
    run('saxpy_cpu_one',saxpy(Backend.CPU,size=65536,iterations=1,core='7'))
elif a.part=='search':
    def probe(n,rep=1):return run('search',saxpy(Backend.NPU,size=n,timeout=600),rep)
    lo,hi=32768,65536
    saved={int(r['params']['size']):r for r in previous if r['label']=='search' and r['ok']}
    if not saved:
        probe(lo);probe(hi)
        saved={int(r['params']['size']):r for r in map(json.loads,path.read_text().splitlines()) if r['label']=='search' and r['ok']}
    # Confirm the first adjacent candidate before binary-searching the interval.
    # The midpoint classifier is a measured log-scale separation, not a guessed threshold.
    cutoff=(float(saved[lo]['fields']['finalize_us'])*float(saved[hi]['fields']['finalize_us']))**.5
    lo=max(n for n,r in saved.items() if float(r['fields']['finalize_us'])<cutoff)
    hi=min(n for n,r in saved.items() if float(r['fields']['finalize_us'])>cutoff)
    if hi-lo>1 and lo+1 not in saved:
        r=probe(lo+1)
        if float(r.fields['finalize_us'])>cutoff:hi=lo+1
        else:lo+=1
    while hi-lo>1:
        mid=(lo+hi)//2;r=probe(mid)
        if float(r.fields['finalize_us'])>cutoff:hi=mid
        else:lo=mid
    (out/'boundary.json').write_text(json.dumps(dict(low=lo,high=hi,cutoff_us=cutoff)))
    # Three fresh independent measurements on EACH side of the adjacent boundary.
    for rep in range(1,4):
        if not any(r['label']=='boundary_low' and r['rep']==rep for r in previous):run('boundary_low',saxpy(Backend.NPU,size=lo,timeout=600),rep)
        if not any(r['label']=='boundary_high' and r['rep']==rep for r in previous):run('boundary_high',saxpy(Backend.NPU,size=hi,timeout=600),rep)
    # Endpoint repetitions for the 13.5 mean and variation, avoiding reuse of a single value.
    if a.repeat_endpoint:
        for rep in range(2,4):
            if not any(r['label']=='saxpy_npu65536' and r['rep']==rep and r['ok'] for r in previous):run('saxpy_npu65536',saxpy(Backend.NPU,size=65536,timeout=600),rep)
else:
    for reps in (1,64,512,2048):
        for rep in range(1,4):
            if not any(r['label']=='startup_npu' and r['params']['reps']==reps and r['rep']==rep for r in previous):run('startup_npu',matmul(Backend.NPU,size=512,reps=reps),rep)
            if not any(r['label']=='startup_gpu' and r['params']['inner_reps']==reps and r['rep']==rep for r in previous):run('startup_gpu',matmul(Backend.GPU,variant='best',size=512,inner_reps=reps),rep)
            time.sleep(.3)
f.close()
