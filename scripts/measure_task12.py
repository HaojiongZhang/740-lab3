"""Collect Task 12's four cases at each size, three process repetitions."""
import argparse, csv, json, time
from pathlib import Path
from profiler import Profiler, matmul, Backend, ThermalReader

ap=argparse.ArgumentParser()
ap.add_argument('--output',default='results/task12')
ap.add_argument('--pause',type=float,default=.25)
ap.add_argument('--max-temp',type=float,default=60)
args=ap.parse_args()
out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
p=Profiler()
thermals=ThermalReader(p.device)
with (out/'measurements.csv').open('w',newline='') as f, (out/'raw.jsonl').open('w') as log:
    w=csv.DictWriter(f,fieldnames=['case','size','rep','execute_us','gflops','finalize_us','internal_cov_pct','fingerprint_sum','fingerprint_frob','shared_impl'])
    w.writeheader()
    for size in (256,512,1024):
        for rep in range(1,4):
            for case in ('sustained','default','baseline','shared'):
                before=thermals.read()
                if not before or max(before.values()) >= args.max_temp:
                    raise RuntimeError(f'Temperature precheck stopped the run: {before}')
                params=dict(size=size,reps=16,perf_mode='default' if case=='default' else 'sustained')
                if case in ('baseline','shared'): params['variant']='spike'
                if case=='shared': params['memshare']='./libmemshare.so'
                r=p.run(matmul(Backend.NPU,**params))
                log.write(json.dumps(dict(case=case,size=size,rep=rep,fields=r.fields,raw=r.raw,error=r.error,temp_c=r.temp_c,temps=r.temps))+'\n');log.flush()
                if r.temp_c is not None and r.temp_c >= args.max_temp:
                    raise RuntimeError(f'Temperature guard: {r.temp_c} C; measurements stopped')
                if not r.ok or not r.compute_us: raise RuntimeError(r.error or r.raw)
                if case=='shared': assert r.fields.get('shared_impl')=='memshare',r.raw
                if case=='baseline': assert r.fields.get('shared_impl')=='none_clientbuf',r.raw
                w.writerow(dict(case=case,size=size,rep=rep,execute_us=r.compute_us,gflops=r.gflops,
                    finalize_us=r.fields['finalize_us'],internal_cov_pct=r.fields['execute_us_cov_pct'],
                    fingerprint_sum=r.fields['fingerprint_sum'],fingerprint_frob=r.fields['fingerprint_frob'],shared_impl=r.fields.get('shared_impl','n/a')))
                f.flush(); time.sleep(args.pause)
            print(f'N={size} repetition {rep} complete',flush=True)
        time.sleep(2)
