"""Validate saved Task 11 measurements and write report/plot (run on laptop)."""
import csv, json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / 'results/task11'
groups = defaultdict(list)
for label in ('baseline','hvx','dma_serial','dma_no_clean','dma_overlap'):
    for row in csv.DictReader((out/f'{label}.csv').open()):
        groups[(label,row['impl'],int(row['p']))].append(row)
assert sum(map(len,groups.values())) == 72
for key, rows in groups.items():
    assert len(rows)==3, key
    if key[0] != 'dma_no_clean':
        expected = {r['checksum'] for r in groups[('dma_overlap','ddr',key[2])]}
        assert len(expected)==1 and {r['checksum'] for r in rows}==expected, key
assert len({r['checksum'] for r in groups[('dma_no_clean','staged',1)]})>1

def vals(key,field): return [float(r[field]) for r in groups[key]]
def mean(key,field='total_us'): return st.mean(vals(key,field))
def cov(key,field='total_us'):
    v=vals(key,field); return 100*st.stdev(v)/st.mean(v) if st.mean(v) else 0
def cell(key,field='total_us'): return f'{mean(key,field):.1f} ({cov(key,field):.2f}%)'

with (out/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['version','impl','P','metric','mean','sample_cov_pct','min','max'])
    for key in groups:
        for field in ('total_us','copy_us','compute_us'):
            v=vals(key,field); w.writerow([*key,field,st.mean(v),cov(key,field),min(v),max(v)])

bw=defaultdict(list)
for rep in range(1,4):
    t=(out/f'probe_{rep}.txt').read_text()
    assert '8388608 bytes) : OK' in t and '8388609 bytes) : FAIL' in t
    assert '2 x   4.0 MB (=   8.0 MB) : first=OK second=OK' in t
    for m in re.finditer(r'^\s+(\d+) KB\s+([\d.]+) GB/s\s+([\d.]+) GB/s\s+([\d.]+) GB/s\s+([\d.]+) GB/s',t,re.M):
        size=int(m[1])
        for metric,value in zip(('VTCM write','VTCM read','DDR write','DDR read'),m.groups()[1:]):
            bw[(size,metric)].append(float(value))
assert len(bw)==24 and all(len(v)==3 for v in bw.values())
sizes=sorted({s for s,m in bw}); metrics=('VTCM write','VTCM read','DDR write','DDR read')
fig,ax=plt.subplots(figsize=(9,5))
for metric in metrics:
    ax.errorbar(sizes,[st.mean(bw[s,metric]) for s in sizes],
                yerr=[st.stdev(bw[s,metric]) for s in sizes],marker='o',capsize=3,label=metric)
ax.set_xscale('log',base=2); ax.set_xticks(sizes,[str(s) for s in sizes])
ax.set_xlabel('Buffer size (KiB)'); ax.set_ylabel('Bandwidth (decimal GB/s)')
ax.set_title('Task 11.3: HVX bandwidth, mean ± sample SD (n=3)')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'bandwidth.png',dpi=170); plt.close(fig)

text='''# Task 11: Move the Data Yourself

Instructions: Lab3_F26 (1).pdf, pages 19–20, Tasks 11.1–11.10.

## Setup, scope, and reproducibility

Measurements were collected on the connected SM8850P board from the lab3 container, with Hexagon SDK 6.7.0.1 and v81 compilation. The supplied updated conv_stage and conv_vtcm_probe clients were pushed from the container's existing build. The initial provisioned probe printed extra reference sections by default; all retained probe data below comes from the updated client's sections 1–4 only. No reference DSP sources, DDR implementation, naive implementation, or stage_kernel were edited. The code change is kernels/lab3/vtcm/student_vtcm.c. The shared DSP library also retains the completed student_morph.c implementation from Task 10.

Three independent dispatches per case; CSV and raw JSONL were flushed after each result. Data size is the harness's 32 MiB (33,554,432 bytes), with sixteen 2 MiB blocks. The final implementation allocates only the provided 4 MiB VTCM grant, split into two 2 MiB buffers, below the board's 8 MiB capacity. P is passes per block. Results use DSP-native time, including exposed copy and compute, not host wall time. Tables show mean (sample CoV %), n=3. summary.csv also reports min/max and CoV for every timing component. Hardware clocks were not modified beyond the supplied clients' behavior: the probe requests its default lab power mode; the stage client does not explicitly request that mode, so cross-client bandwidth comparisons describe observed paths rather than an exactly frequency-matched hardware ceiling.

The installed DMA API constrains descriptor width/height/stride to 16 bits. Therefore each 2 MiB transfer uses a 2D descriptor with 64 rows, width/stride 32,768 bytes, source cache bypass, and VTCM destination. A naive 2 MiB 1D descriptor would exceed the API limit. Descriptor setup/submission errors return a failure, never a memcpy fallback. Each descriptor remains valid until DMA completion. Cache cleaning is enabled in the final code. Experimental source snapshots are in versions/; the intentionally incorrect no-clean variant is not the final source.

## 11.1: capacity

All three probes report one page of 8,388,608 bytes: **8 MiB**. A single 8 MiB acquire succeeds; 8 MiB + 1 byte, 8 MiB + 4096 bytes, 12 MiB, and 16 MiB fail. Two concurrent 4 MiB acquires both succeed, reaching the same 8 MiB total. For two 5 MiB requests, the first succeeds and the second fails. Thus the advertised capacity holds for the tested allocations. Complete acquisition results and times are in probe_1.txt through probe_3.txt.

## 11.2: VTCM bandwidth and timing

Section 4 results follow. The columns are mean GB/s (sample CoV %) from the probe's printed values, rounded by that client to 0.1 GB/s.

| Buffer (KiB) | VTCM write | VTCM read | Ordinary write | Ordinary read |
|---|---:|---:|---:|---:|
'''
for size in sizes:
    text+=f'| {size} | '+' | '.join(f'{st.mean(bw[size,m]):.2f} ({100*st.stdev(bw[size,m])/st.mean(bw[size,m]):.2f}%)' for m in metrics)+' |\n'
text+='''
In dsp_conv_vtcm_probe.c, modes 3/4 acquire VTCM, lock HVX, and perform an untimed write to populate the buffer. The timer then surrounds only bw_write or bw_read for `iters` passes; acquisition, HVX locking, initialization, unlock, and release are excluded. Each size moves 268,435,456 bytes in the timed direction: iters = 268435456 / size. The same loops operate on aligned ordinary memory in modes 5/6. Bandwidth is size × iters / elapsed time.

bw_read uses four independent HVX accumulators, combines their vector sums, reduces to a scalar, and returns it through the probe output. The observable checksum prevents dead-load elimination; four accumulators also reduce the dependency-chain bottleneck of a single running sum. Writes issue 128-byte vector stores. These are warmed-buffer microbenchmarks, not uncached DRAM-only measurements.

## 11.3: cache transition

![VTCM and ordinary-memory bandwidth](bandwidth.png)

Ordinary read bandwidth is high at 64/256 KiB, falls at 1 MiB, and drops sharply at 2 MiB to roughly 14 GB/s; 4/8 MiB remain around 11–13 GB/s. This brackets the effective cache-resident working set near the 1 MiB scale and indicates that 2 MiB no longer fits effectively. The sweep does not determine an exact architectural cache capacity or associativity. VTCM does not show this cache-capacity collapse because the allocation remains in its explicit scratchpad. Ordinary write figures differ from reads because stores, buffering, and cache behavior differ.

## 11.4: convolution weights

Weight bytes = 36 C² for K=3 and float32 weights.

| C | Bytes | KiB | MiB |
|---|---:|---:|---:|
| 64 | 147,456 | 144 | 0.140625 |
| 128 | 589,824 | 576 | 0.5625 |
| 256 | 2,359,296 | 2304 | 2.25 |

C=64 and C=128 weights are below the observed read-bandwidth cliff and may already be served well by cache, though activations and other working data can compete with them. C=256 weights exceed that effective cache range, so repeatedly reusing them from VTCM is more likely to repay the initial copy and avoid repeated cache misses. All three sizes fit the measured VTCM capacity in isolation. Benefit depends on reuse and the rest of the working set, not size alone.

## 11.5–11.7: serial-copy experiments, P=1

The baselines below are the initially measured supplied DDR/naive paths. The remaining rows use the student staged path.

| Version | Total us (CoV) | Copy us (CoV) | Compute us (CoV) | Copy GB/s | Checksum |
|---|---:|---:|---:|---:|---|
'''
keys=[('baseline','ddr',1),('baseline','naive',1),('hvx','staged',1),('dma_serial','staged',1),('dma_no_clean','staged',1)]
for key in keys:
    c=mean(key,'copy_us'); checks=','.join(sorted({r['checksum'] for r in groups[key]}))
    text+=f'| {key[0]} / {key[1]} | {cell(key)} | {cell(key,"copy_us")} | {cell(key,"compute_us")} | {33554432/(c*1000) if c else 0:.2f} | {checks} |\n'
base=mean(('baseline','ddr',1)); hvx=mean(('hvx','staged',1)); ser=mean(('dma_serial','staged',1))
text+=f'''
**11.5.** Direct DDR has no explicit copy. Naive staging spends almost all its time in memcpy; its roughly 0.51 GB/s payload copy rate is far below both the ordinary-memory HVX read rate and VTCM bandwidth. This shows that the memcpy path is the bottleneck, not VTCM's inherent storage speed. Payload copy bandwidth counts bytes copied once, whereas the probe times one read or write direction; actual copying also performs destination writes.

**11.6.** my_copy now loads/stores HVX_Vector values (128 bytes per vector) and handles a byte tail. HVX staged total is {hvx:.1f} us versus DDR {base:.1f} us, a {base/hvx:.2f}x speedup. Even with no reuse it benefits from the faster VTCM compute path, although the serial HVX copy still adds several milliseconds.

**11.7.** Coherent serial DMA total is {ser:.1f} us; its transfer uses the DMA engine with source-cache cleaning, followed by wait-for-idle before compute. Removing cleaning reduces apparent copy/total time but returns incorrect, changing checksums instead of the correct P=1 value 2,097,104. Dirty source cache lines have not necessarily reached the backing memory read by cache-bypassing DMA. The explicitly requested `--repeat 3` run is saved in dma_no_clean_repeat3.txt; it reports distinct fingerprints and therefore cannot be accepted as a speedup. Cleaning was restored before the final builds.

## 11.8: double-buffered DMA sweep

The next transfer targets the other buffer and is submitted before computing the current buffer. DMA completion is awaited before that destination becomes the next input. The first block is copied and awaited before entering the pipeline. copy_us counts the initial transfer, source cache cleaning/descriptor submission, and exposed completion waits; it does not count background transfer time twice. compute_us is elapsed time around stage_kernel. Bookkeeping and HVX lock/unlock account for the small difference between total and copy+compute.

All three implementations below have identical checksums for a given P in all three repetitions. Serial DMA (11.7's algorithm) was additionally measured at every P, allowing a direct same-P overlap comparison.

| P | DDR total us (CoV) | DDR copy us | Naive total us (CoV) | Staged total us (CoV) | Staged copy us (CoV) | Staged compute us (CoV) | Serial DMA total us | Serial copy us | DDR/staged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
for passes in (1,2,4,8,16):
    d=('dma_overlap','ddr',passes); n=('dma_overlap','naive',passes); s=('dma_overlap','staged',passes); serial=('dma_serial','staged',passes)
    text+=f'| {passes} | {cell(d)} | 0 | {cell(n)} | {cell(s)} | {cell(s,"copy_us")} | {cell(s,"compute_us")} | {mean(serial):.1f} | {mean(serial,"copy_us"):.1f} | {mean(d)/mean(s):.2f}x |\n'
text+='\nExposed-copy reduction relative to the serial algorithm (an operational estimate, since these are separate runs):\n\n| P | Copy time removed (us) | Serial copy fraction removed | Total time saved (us) |\n|---|---:|---:|---:|\n'
for passes in (1,2,4,8,16):
    a=('dma_serial','staged',passes); b=('dma_overlap','staged',passes)
    hidden=mean(a,'copy_us')-mean(b,'copy_us')
    text+=f'| {passes} | {hidden:.1f} | {100*hidden/mean(a,"copy_us"):.1f}% | {mean(a)-mean(b):.1f} |\n'
d=('dma_overlap','ddr',1); s=('dma_overlap','staged',1)
text+=f'''
Not all copy costs disappear. The first transfer has no preceding computation to hide behind, source-cache cleaning and submission run on the DSP thread before computing, and any DMA longer than the concurrent compute leaves an exposed wait. For the last block there is no following transfer to overlap; its computation still has to complete. Increasing P gives DMA more computation to hide behind, but it cannot hide serial setup/cleaning or pipeline startup. Ratios and reductions here are based on means, not an assertion that the separately measured serial and overlapping transfers have identical instantaneous bandwidth.

## 11.9: why staging wins at P=1

At P=1, DDR total is {mean(d):.1f} us, whereas overlapped staging takes {mean(s):.1f} us ({mean(d)/mean(s):.2f}x faster). DDR compute itself takes {mean(d,'compute_us'):.1f} us; VTCM compute takes {mean(s,'compute_us'):.1f} us, with only {mean(s,'copy_us'):.1f} us of exposed copy cost. This gain does not require reusing each block: DMA streams the source into VTCM while HVX consumes a different block, and VTCM feeds the vector compute faster than direct ordinary-memory accesses. The saved compute/memory-stall time exceeds the remaining copy cost. The naive memcpy version demonstrates why merely placing data in VTCM is insufficient.

## 11.10: corresponding CPU mechanisms

VTCM performs explicit placement like a CPU cache; DMA plus pipelining performs explicit data movement ahead of use like a hardware prefetcher/cache-fill machinery. Manual control gives predictable residency, avoids unwanted cache eviction, controls transfer size and timing, and can overlap memory movement with useful work. Its costs are limited scratchpad space, extra copies and startup, explicit coherence and synchronization, more complex code, and workload-specific tuning. Hardware caches/prefetchers adapt automatically and preserve the ordinary coherent load/store abstraction, but provide less direct control over residency and movement.

## Validation and reproduction

The correct baseline, HVX, serial-DMA, and final-overlap datasets all match the DDR checksum at the same P; 69 correct dispatches were checked. The three additional no-clean dispatches intentionally fail correctness, as do the separate requested CLI repeat run. Raw output, component timings, variants, and probe logs are retained. No checksum comparison assumes linearity in P: float accumulation/rounding can make that assumption invalid at large P.

Final coherent/overlapped code is in kernels/lab3/vtcm/student_vtcm.c with COPY_MODE=2, DMA_CLEAN=1, OVERLAP=1. Preserve Task 10 in the shared library by building inside the container with:

```bash
bash scripts/build_student.sh vtcm --exclude dsp_conv_morph_imp.c --add /workspace/kernels/lab3/morph/student_morph.c
PYTHONPATH=/workspace python scripts/measure_task11.py probe --probe
PYTHONPATH=/workspace python scripts/measure_task11.py dma_overlap --impls ddr,naive,staged --passes 1,2,4,8,16
```

The supplied script regenerates incremental CSV/JSONL; use a new label to retain previous results. For serial/HVX/no-clean experiments, copy the corresponding versions/*.c snapshot into the container's student_vtcm.c, rebuild using the same command, then invoke the runner with that variant's label. Always restore the final coherent source afterwards. Results/plots can be regenerated on the laptop with `python scripts/report_task11.py` (matplotlib required for the plot).

The measured final DSP library SHA-256, verified equal on the board and in the container, is 342084babf775373fd5b89cfb7774af5f1651c5a845ea766f79558d3e5dad16d. Afterwards only explanatory comments in student_vtcm.c were cleaned up; executable statements are unchanged. SDK/reference files were read only. Existing generated client/build outputs in the container were rebuilt as needed; source changes on the laptop are limited to the Task 11 student file and the previously completed Task 10 file, plus new measurement/report scripts and results.
'''
(out/'report.md').write_text(text,encoding='utf-8')
print('Validated 72 dispatches (69 correct, 3 intentionally incoherent); wrote report.md, summary.csv, bandwidth.png.')
