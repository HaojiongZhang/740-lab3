"""Generate all Task 13 answers from the retained raw measurements."""
import csv,json,math,statistics as st
from pathlib import Path
from collections import defaultdict

root=Path(__file__).resolve().parents[1];out=root/'results/task13'
read=lambda name:[json.loads(s) for s in (out/name).read_text().splitlines()]
main=read('main.jsonl');startup=read('startup.jsonl');search=read('search.jsonl')
good=[r for r in search if r['ok']];bad=[r for r in search if not r['ok']]
prior=list(csv.DictReader((root/'results/task12_redo/measurements.csv').open()))
boundary=json.loads((out/'boundary.json').read_text());lo=boundary['low'];hi=boundary['high']
assert hi==lo+1
assert len(main)==27 and len(startup)==24
for label in ('boundary_low','boundary_high'):assert sum(r['label']==label for r in good)==3
g=defaultdict(list)
for r in main:g[r['label'],int(r['params']['size'])].append(r)
for r in good:g[r['label'],int(r['params']['size'])].append(r)
def m(label,n,field=None):
    return st.mean(float(r['fields'][field] if field else r['per_iter_us']) for r in g[label,n])
def cv(label,n,field=None):
    v=[float(r['fields'][field] if field else r['per_iter_us']) for r in g[label,n]]
    return 100*st.stdev(v)/st.mean(v) if len(v)>1 and st.mean(v) else 0
def old(case,n,field='execute_us'):
    return st.mean(float(r[field]) for r in prior if r['case']==case and int(r['size'])==n)
def kstrict(setup,fast,slow):
    return max(1,math.floor(setup/(slow-fast))+1) if slow>fast else None
npu=old('sustained',512);finalize=old('sustained',512,'finalize_us');gpu=m('gpu512',512)
k9=kstrict(finalize,npu,gpu)
def fit(rows):
    x=[int(r['params'].get('reps',r['params'].get('inner_reps'))) for r in rows]
    y=[float(r['fields']['host_wall_us']) for r in rows];xm=st.mean(x);ym=st.mean(y)
    slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x)
    intercept=ym-slope*xm
    residual=[b-(intercept+slope*a) for a,b in zip(x,y)]
    return intercept,slope,(st.mean(z*z for z in residual))**.5
fn=fit([r for r in startup if r['label']=='startup_npu']);fg=fit([r for r in startup if r['label']=='startup_gpu'])
kfit=kstrict(fn[0]-fg[0],fn[1],fg[1]);kfixed=kstrict(fn[0]-fg[0],npu,gpu)
with (out/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['label','size','n','mean_execute_us','sample_cov_pct','mean_finalize_us'])
    for (lab,n),rs in g.items():w.writerow([lab,n,len(rs),m(lab,n),cv(lab,n),m(lab,n,'finalize_us') if 'finalize_us' in rs[0]['fields'] else ''])
text='''# Task 13: Two Knobs, the Compiler, and the Choice

Instructions: Lab3_F26 (1).pdf, Tasks 13.1–13.10, pages 22–23.

## Scope, method, and correctness

No lab kernel, profiler, SDK, or reference-client source was modified for Task 13. New scripts collect, independently validate, and report the measurements. Part 12 comparisons use the completed cool-board **results/task12_redo** dataset. main.jsonl preserves the earlier successful INT8/CPU/GPU/finalization measurements. Each specified three-repeat benchmark has three independent processes; the finalization check has five. CPU totals are divided by their iteration counts. NPU execute_us_mean is already per execution. GPU duration_ns is the entire timed inner_reps loop and is divided by 1000×inner_reps for per-matmul time; the report derives this from raw fields even where an early runner's per_iter_us field was not yet normalized. CoV means sample CoV across process means unless described as the client's internal CoV.

The final startup experiment uses direct Windows adb consistently for all 24 processes; partial container-proxy startup data is retained in startup_container_partial.jsonl and startup_container_snapshot.jsonl but excluded from fits. host_wall_us includes adb transport and process startup/teardown, so its intercept is an operational whole-process/transport cost, not a pure accelerator initialization counter. No experiment from Task 13 was mixed into the Task 12 raw dataset.

The supplied CPU matmul uses [-1,1) inputs from one advancing seed stream; GPU/NPU matmul use [0,1) streams seeded separately. INT8 uses its own signed quantized inputs and output scale 4. Thus CPU/INT8 versus FP16 speed comparisons are the requested fixed-shape benchmark comparisons, not proof of numerically interchangeable inputs. Repeats have stable fingerprints within each configuration. GPU and NPU FP16-related fingerprints agree relatively within 0.001%; large GPU inner loops accumulate rounding, checked within 0.03%. CPU SAXPY's 50,000 iterations update y in place, whereas the NPU repeats graph executions with fixed x/y; its final fingerprint is therefore not expected to equal a one-pass fingerprint. A one-pass CPU check was separately recorded.

validation.json records an independent NumPy implementation of the supplied splitmix64 input stream, FP16 input conversion, alpha=2 arithmetic, output rounding, sum and Frobenius norm. Every successful SAXPY search/boundary result agrees within 0.00002 absolute for each fingerprint component. INT8 verification is harness success, deployed-client hash equivalence to the supplied binary, and stable fingerprints, not an independently established quantizer accuracy bound. Failed supplemental attempts are preserved and excluded from timing estimates.

## 13.1: prediction and the two advantages

Initial prediction: changing performance mode should matter most when small kernels are dominated by fixed overhead and wake/dispatch behavior; narrowing tensors should become more useful as payload sizes grow.

| N | FP16 sustained us (Task 12) | FP16 default us (Task 12) | INT8 sustained us | INT8 process CoV | Power-mode speedup | INT8 speedup |
|---|---:|---:|---:|---:|---:|---:|
'''
for n in (256,512,1024):
    a=old('sustained',n);b=old('default',n);c=m('int8',n)
    text+=f'| {n} | {a:.2f} | {b:.2f} | {c:.2f} | {cv("int8",n):.2f}% | {b/a:.3f}x | {a/c:.3f}x |\n'
text+='''
Power-mode advantage shrinks with N, whereas INT8 advantage grows over these sizes, consistent with the prediction. Halving tensor bytes cannot halve fixed host/RPC dispatch, synchronization, setup inside execute, or all hardware arithmetic/layout work. Quantized execution also has scaling/requantization work. Only part of FP16 elapsed time is removable by reduced payload size, so the measured gain is far below 2x. These are default-client INT8 benchmarks, not the Task 12 shared-memory path.

## 13.2: product choice

For the measured latency-focused workload I would choose sustained performance mode: its speedup is much larger throughout the measured range and preserves the FP16 data representation. The tradeoff is its potential power/thermal cost and reduced suitability for a sustained thermally constrained product. Choosing it alone gives up INT8's storage/traffic reduction and additional speedup. INT8 is attractive where quantization accuracy can be validated and memory/energy constraints dominate. These timing experiments do not measure energy or establish a product-wide accuracy bound.

## 13.3: finalization is real compilation work

Five fresh FP16 N=512 processes were measured. Each value is retained below.

| Process | Finalize us | Execute mean us |
|---|---:|---:|
'''
for r in g['finalize512',512]:text+=f'| {r["rep"]} | {r["fields"]["finalize_us"]} | {r["per_iter_us"]:.2f} |\n'
text+=f'''
Finalization mean is **{m('finalize512',512,'finalize_us'):.2f} us**, sample CoV **{cv('finalize512',512,'finalize_us'):.2f}%**. Task 12 sustained finalization means for N=256,512,1024 are {old('sustained',256,'finalize_us'):.2f}, {finalize:.2f}, {old('sustained',1024,'finalize_us'):.2f} us. It is repeatable relative to its scale and grows with graph size/shape; it is genuine graph optimization/code-generation work, with ordinary timing noise superimposed. It should not be dismissed as random system noise.

## 13.4: amortizing finalization against CPU matmul

For K uses of one finalized graph, CPU total is K×C and NPU total is F+K×E. NPU strictly wins when **K > F/(C−E)**, provided C>E. The smallest integer is floor(F/(C−E))+1. If C≤E and F≥0 there is no positive break-even. This section intentionally uses the handout's finalize-only model, excluding other startup costs.

Use Task 12 N=512 F={finalize:.2f} us and E={npu:.2f} us:

| CPU configuration | Iterations/process | Mean us/matmul | Process CoV | F/(C−E) | First winning K |
|---|---:|---:|---:|---:|---:|
'''
for lab,it in [('cpu1',40),('cpu8',150)]:
    c=m(lab,512);text+=f'| {lab} | {it} | {c:.2f} | {cv(lab,512):.2f}% | {finalize/(c-npu):.3f} | {kstrict(finalize,npu,c)} |\n'
endpoint=next(r for r in good if r['label']=='search' and r['params']['size']==65536)
se=float(endpoint['fields']['execute_us_mean']);sf=float(endpoint['fields']['finalize_us']);sc=m('saxpy_cpu',65536)
text+=f'''
CPU8 is pinned to all eight CPUs with eight workers; CPU1 uses Prime core 7 with one worker. No total time was mistaken for one iteration. Because the supplied CPU/NPU input generation and precision differ, these K values describe the requested arithmetic-shape implementations; they are not a promise of a drop-in numerical substitution.

## 13.5: SAXPY has no break-even here

The 65,536-element CPU benchmark takes **{sc:.5f} us/pass**, process CoV {cv('saxpy_cpu',65536):.2f}%, over 50,000 passes per process and three processes. The successful NPU endpoint process reports **{se:.2f} us/execute**, internal five-execution CoV {endpoint['fields']['execute_us_cov_pct']:.3f}%, with finalization **{sf:.0f} us**. Here C−E is negative ({sc-se:.2f} us). No positive K repays finalization: even once F is fully amortized, the NPU execute is slower than CPU SAXPY. The short vector's cheap arithmetic does not compensate for NPU boundary/dispatch work. The endpoint has one retained successful process with five internal timed executions; no between-process CoV is claimed. A later optional endpoint-repeat attempt failed without a valid completion record and was rejected rather than treated as a measurement.

## 13.6: exact compiler cliff and complete search order

The original unrestricted endpoints were 32,768 and 65,536 elements. The first successful recorded finalizations were {good[0]['fields']['finalize_us']} us and {endpoint['fields']['finalize_us']} us. Earlier exploratory CLI probes, before the saved search, observed 274,685 us and 40,336,441 us respectively; those exploratory values are disclosed but not used in repeated statistics.

Heat management: unrestricted long compilation rapidly heated Prime cores, so aborted attempts were not accepted. The remaining compiler search was restricted to Gold cores (taskset mask 0x3f), first capped at 2.112 GHz and then 3.0336 GHz. Before each resumed run, relevant sensors had to be ≤38°C; during long runs, telemetry was sampled approximately every 3 seconds and the workload was stopped at 58°C. Post-run pauses were added. CPU limits were restored afterwards to min 787,200/max 3,628,800 kHz. These caps change absolute compiler/RPC timing, so the comparison of adjacent sizes below uses only the same final capped configuration. NPU sustained mode was unchanged. Initial unrestricted points and earlier capped points are separately identified in the trace.

| Order | Label | Elements | Finalize us | Execute us | Setup |
|---|---|---:|---:|---:|---|
'''
for i,r in enumerate(search,1):
    setup=('Gold cap '+str(r.get('compiler_cap_khz') or 2112000)+' kHz') if r.get('cool_compiler') else ('Gold cap 2112000 kHz' if r.get('temp_peak_c') is not None else 'original unrestricted')
    if not r['ok']:text+=f'| {i} | {r["label"]} FAILED / excluded | {r["params"]["size"]} | — | — | {setup} |\n'
    else:text+=f'| {i} | {r["label"]} | {r["params"]["size"]} | {r["fields"]["finalize_us"]} | {r["fields"]["execute_us_mean"]:.2f} | {setup} |\n'
lf=m('boundary_low',lo,'finalize_us');hf=m('boundary_high',hi,'finalize_us');le=m('boundary_low',lo);he=m('boundary_high',hi)
text+=f'''
The isolated adjacent transition is **{lo:,} → {hi:,} elements**: the first large cliff identified in this bracket. Each side was then measured in three fresh processes under the same Gold cap. Larger sizes also show additional compiler cost/growth; this experiment does not claim that no other discontinuities exist. The threshold classifier was the geometric mean of measured original endpoint finalize times ({boundary['cutoff_us']:.0f} us), which cleanly separates the ~1.4 s and ~13 s adjacent cases even under the cap. The exact adjacency is demonstrated by measurements, not extrapolation from a sparse graph.

## 13.7: compilation changes much more than execution

| Side | Elements | Finalize mean us (sample CoV) | Execute mean us (sample CoV) |
|---|---:|---:|---:|
| Below | {lo} | {lf:.2f} ({cv('boundary_low',lo,'finalize_us'):.2f}%) | {le:.2f} ({cv('boundary_low',lo):.2f}%) |
| Above | {hi} | {hf:.2f} ({cv('boundary_high',hi,'finalize_us'):.2f}%) | {he:.2f} ({cv('boundary_high',hi):.2f}%) |

One extra element changes finalization by **{hf/lf:.3f}x** (+{hf-lf:.2f} us), but execution by only **{he/le:.3f}x** (+{he-le:.2f} us). This points to a discrete compiler lowering/tiling/optimization decision or search complexity at a shape boundary, not proportional growth of the requested SAXPY arithmetic. The saved high-side diagnostics contain many ConvLayer lowering cost warnings; that supports a changed compiler path but does not prove a particular internal algorithm. The independent FP16 oracle confirms that successful outputs remain correct on both sides.

## 13.8: dynamically changing shapes

Dynamic shapes can trigger unpredictable compilation latency, memory consumption and startup stalls when crossing a compiler boundary, even when execution time changes little. Two approaches are:

1. Precompile/cache a bounded collection of expected shape graphs and reuse them. This moves compilation out of request latency but costs storage/memory, warm-up time, and a cache-miss policy for unforeseen shapes.
2. Bucket/pad requests to a small set of validated shapes, ideally avoiding an expensive compiler region. This reduces the number of graphs and makes latency predictable, but wastes padded arithmetic/bandwidth and can add batching delay. Padding and slicing must preserve semantics.

For rare or latency-critical cold shapes, a CPU/GPU fallback is also useful; that adds implementation and routing complexity. None of these strategies assumes that all possible shapes can be cached for free.

## 13.9: GPU versus NPU, finalize-only assumption

GPU best at N=512 averages **{gpu:.2f} us**, process CoV {cv('gpu512',512):.2f}%; Task 12 sustained NPU averages **{npu:.2f} us**. NPU is faster per execution. Under the requested assumption that only NPU finalization costs startup and GPU startup is zero:

`F + K*E_NPU < K*E_GPU`, hence `K > {finalize:.2f}/({gpu:.2f}−{npu:.2f}) = {finalize/(gpu-npu):.3f}`.

The first winning integer is **K={k9}**. This is a deliberately incomplete startup model, corrected next.

## 13.10: full-process startup experiment and engine choice

Using only profiler fields, run each engine in fresh processes with R=1,64,512,2048 internal repetitions, three processes per R, alternating engines. Record host_wall_us (the same field shown by single-run `-v`) and native execution timing. Fit **H(R)=S+R×E** separately by ordinary least squares. Varying R separates roughly fixed cost from repeated execution rather than simply treating all host wall time as setup. NPU uses reps=R; GPU uses inner_reps=R. GPU R operations occur inside one kernel and average their result, so this model describes that provided repetition mechanism, not R independent GPU launches.

| Engine | R | Mean host wall us | Host-wall sample CoV | Native mean us/logical matmul |
|---|---:|---:|---:|---:|
'''
for label in ('startup_npu','startup_gpu'):
    for count in (1,64,512,2048):
        rs=[r for r in startup if r['label']==label and int(r['params'].get('reps',r['params'].get('inner_reps')))==count]
        hs=[float(r['fields']['host_wall_us']) for r in rs]
        native=st.mean(float(r['fields']['execute_us_mean']) if label=='startup_npu' else float(r['fields']['duration_ns'])/1000/count for r in rs)
        text+=f'| {label} | {count} | {st.mean(hs):.2f} | {100*st.stdev(hs)/st.mean(hs):.2f}% | {native:.2f} |\n'
text+=f'''
| Fit | Fixed S (us) | Slope E (us/operation) | Residual RMSE (us) |
|---|---:|---:|---:|
| NPU | {fn[0]:.2f} | {fn[1]:.3f} | {fn[2]:.2f} |
| GPU | {fg[0]:.2f} | {fg[1]:.3f} | {fg[2]:.2f} |

Both fixed costs are substantial. They include backend/runtime/context initialization, graph compilation or OpenCL program setup, allocation, input initialization/transfer, warm-up where present, output readback/checksum, process teardown, and adb/transport overhead. They are not just NPU graphFinalize. GPU has setup too even though its event timer reports none of it. Cache state and transport noise remain; the RMSE and process CoVs show the uncertainty rather than implying microsecond-level precision in a fixed intercept.

The fitted total-time comparison is `S_NPU+K*E_NPU < S_GPU+K*E_GPU`. Therefore **K > ({fn[0]:.2f}−{fg[0]:.2f})/({fg[1]:.3f}−{fn[1]:.3f})**, giving first winning **K={kfit}**. This is **{kfit-k9} more operations** than the finalize-only K={k9} model. If the fitted startup estimates are combined instead with the single-operation execution means from 13.9, the first winning K is **{kfixed}**; this alternative makes the sensitivity to different repetition mechanisms explicit. GPU per-operation timing increases in the long inner loop relative to R=1, so the two K models should not be silently treated as identical. These are approximate workload/transport-specific estimates, not universal hardware constants.

I would deploy the NPU for a persistent, repeatedly reused FP16 graph at this size, where its lower execution time can amortize initialization. For an occasional cold invocation under this measured process/transport setup, the GPU has lower fixed overhead and can finish sooner. Numerical precision, sustained thermal behavior, and application energy requirements still need product-specific evaluation; the supplied CPU/INT8 input differences prevent claiming all engines are exact numerical substitutes. Shared buffers from Task 12 could further improve NPU execution, but were not substituted into this handout comparison.

## Reproduction and verification artifacts

Set the project's platform-tools directory on PATH and PYTHONPATH to the repository root. Then use:

```text
python scripts/measure_task13.py main
python scripts/measure_task13.py startup
python scripts/measure_task13.py search --cool-compiler
python scripts/validate_task13.py
python scripts/report_task13.py
```

The runner appends records and resumes startup/search checkpoints; archive a log before a deliberate fresh rerun. NumPy is needed only for the independent validator. The CPU cap is restored by the runner's exit handler; the final observed original limits were confirmed separately. The optional --repeat-endpoint flag requests extra expensive 65,536-element process repetitions; it is not needed for the handout's single NPU SAXPY process, which already averages five internal executions. One such supplemental attempt returned no valid completion and is recorded as FAILED/excluded above. thermal_stops.jsonl contains an aborted unrestricted attempt; its high temperatures are not included in accepted measurements. The original earlier interrupted container search checkpoint is preserved in the successful trace. No failed, aborted, or partial-container startup timing enters any reported mean or fit.
'''
(out/'report.md').write_text(text,encoding='utf-8')
(out/'break_even.json').write_text(json.dumps(dict(cpu1=kstrict(finalize,npu,m('cpu1',512)),cpu8=kstrict(finalize,npu,m('cpu8',512)),saxpy=None,gpu_finalize_only=k9,gpu_full_fit=kfit,gpu_full_fixed_execution=kfixed,fit_npu=fn,fit_gpu=fg),indent=2))
print(f'Wrote complete report: boundary {lo}->{hi}, GPU K={k9} finalize-only, K={kfit} full fitted process model.')
