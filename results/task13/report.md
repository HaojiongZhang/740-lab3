# Task 13: Two Knobs, the Compiler, and the Choice

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
| 256 | 74.20 | 737.57 | 67.63 | 2.29% | 9.940x | 1.097x |
| 512 | 131.83 | 982.50 | 109.37 | 0.54% | 7.453x | 1.205x |
| 1024 | 397.47 | 1724.07 | 286.03 | 1.30% | 4.338x | 1.390x |

Power-mode advantage shrinks with N, whereas INT8 advantage grows over these sizes, consistent with the prediction. Halving tensor bytes cannot halve fixed host/RPC dispatch, synchronization, setup inside execute, or all hardware arithmetic/layout work. Quantized execution also has scaling/requantization work. Only part of FP16 elapsed time is removable by reduced payload size, so the measured gain is far below 2x. These are default-client INT8 benchmarks, not the Task 12 shared-memory path.

## 13.2: product choice

For the measured latency-focused workload I would choose sustained performance mode: its speedup is much larger throughout the measured range and preserves the FP16 data representation. The tradeoff is its potential power/thermal cost and reduced suitability for a sustained thermally constrained product. Choosing it alone gives up INT8's storage/traffic reduction and additional speedup. INT8 is attractive where quantization accuracy can be validated and memory/energy constraints dominate. These timing experiments do not measure energy or establish a product-wide accuracy bound.

## 13.3: finalization is real compilation work

Five fresh FP16 N=512 processes were measured. Each value is retained below.

| Process | Finalize us | Execute mean us |
|---|---:|---:|
| 1 | 4947 | 131.60 |
| 2 | 5270 | 134.10 |
| 3 | 4946 | 135.40 |
| 4 | 5110 | 134.60 |
| 5 | 4968 | 133.10 |

Finalization mean is **5048.20 us**, sample CoV **2.80%**. Task 12 sustained finalization means for N=256,512,1024 are 4427.00, 5049.33, 9566.33 us. It is repeatable relative to its scale and grows with graph size/shape; it is genuine graph optimization/code-generation work, with ordinary timing noise superimposed. It should not be dismissed as random system noise.

## 13.4: amortizing finalization against CPU matmul

For K uses of one finalized graph, CPU total is K×C and NPU total is F+K×E. NPU strictly wins when **K > F/(C−E)**, provided C>E. The smallest integer is floor(F/(C−E))+1. If C≤E and F≥0 there is no positive break-even. This section intentionally uses the handout's finalize-only model, excluding other startup costs.

Use Task 12 N=512 F=5049.33 us and E=131.83 us:

| CPU configuration | Iterations/process | Mean us/matmul | Process CoV | F/(C−E) | First winning K |
|---|---:|---:|---:|---:|---:|
| cpu1 | 40 | 9842.42 | 0.06% | 0.520 | 1 |
| cpu8 | 150 | 1702.08 | 1.60% | 3.216 | 4 |

CPU8 is pinned to all eight CPUs with eight workers; CPU1 uses Prime core 7 with one worker. No total time was mistaken for one iteration. Because the supplied CPU/NPU input generation and precision differ, these K values describe the requested arithmetic-shape implementations; they are not a promise of a drop-in numerical substitution.

## 13.5: SAXPY has no break-even here

The 65,536-element CPU benchmark takes **10.21465 us/pass**, process CoV 0.14%, over 50,000 passes per process and three processes. The successful NPU endpoint process reports **452.60 us/execute**, internal five-execution CoV 2.136%, with finalization **41439596 us**. Here C−E is negative (-442.39 us). No positive K repays finalization: even once F is fully amortized, the NPU execute is slower than CPU SAXPY. The short vector's cheap arithmetic does not compensate for NPU boundary/dispatch work. The endpoint has one retained successful process with five internal timed executions; no between-process CoV is claimed. A later optional endpoint-repeat attempt failed without a valid completion record and was rejected rather than treated as a measurement.

## 13.6: exact compiler cliff and complete search order

The original unrestricted endpoints were 32,768 and 65,536 elements. The first successful recorded finalizations were 270070 us and 41439596 us. Earlier exploratory CLI probes, before the saved search, observed 274,685 us and 40,336,441 us respectively; those exploratory values are disclosed but not used in repeated statistics.

Heat management: unrestricted long compilation rapidly heated Prime cores, so aborted attempts were not accepted. The remaining compiler search was restricted to Gold cores (taskset mask 0x3f), first capped at 2.112 GHz and then 3.0336 GHz. Before each resumed run, relevant sensors had to be ≤38°C; during long runs, telemetry was sampled approximately every 3 seconds and the workload was stopped at 58°C. Post-run pauses were added. CPU limits were restored afterwards to min 787,200/max 3,628,800 kHz. These caps change absolute compiler/RPC timing, so the comparison of adjacent sizes below uses only the same final capped configuration. NPU sustained mode was unchanged. Initial unrestricted points and earlier capped points are separately identified in the trace.

| Order | Label | Elements | Finalize us | Execute us | Setup |
|---|---|---:|---:|---:|---|
| 1 | search | 32768 | 270070 | 174.80 | original unrestricted |
| 2 | search | 65536 | 41439596 | 452.60 | original unrestricted |
| 3 | search | 32769 | 287518 | 179.00 | original unrestricted |
| 4 | search | 49152 | 531991 | 229.00 | original unrestricted |
| 5 | search | 57344 | 653850 | 255.00 | original unrestricted |
| 6 | search | 61440 | 737001 | 261.80 | original unrestricted |
| 7 | search | 63488 | 41672197 | 414.20 | original unrestricted |
| 8 | search | 62464 | 108490918 | 421.60 | Gold cap 2112000 kHz |
| 9 | search | 61952 | 102176248 | 420.60 | Gold cap 2112000 kHz |
| 10 | search | 61441 | 1364821 | 283.20 | Gold cap 3033600 kHz |
| 11 | search | 61696 | 70698146 | 414.40 | Gold cap 3033600 kHz |
| 12 | search | 61568 | 13240344 | 277.00 | Gold cap 3033600 kHz |
| 13 | search | 61504 | 1366674 | 278.20 | Gold cap 3033600 kHz |
| 14 | search | 61536 | 13277011 | 288.80 | Gold cap 3033600 kHz |
| 15 | search | 61520 | 13241521 | 289.40 | Gold cap 3033600 kHz |
| 16 | search | 61512 | 13262755 | 291.20 | Gold cap 3033600 kHz |
| 17 | search | 61508 | 13285363 | 289.20 | Gold cap 3033600 kHz |
| 18 | search | 61506 | 13253686 | 290.40 | Gold cap 3033600 kHz |
| 19 | search | 61505 | 13227602 | 288.00 | Gold cap 3033600 kHz |
| 20 | boundary_low | 61504 | 1390893 | 276.40 | Gold cap 3033600 kHz |
| 21 | boundary_high | 61505 | 13242229 | 290.60 | Gold cap 3033600 kHz |
| 22 | boundary_low | 61504 | 1360138 | 277.60 | Gold cap 3033600 kHz |
| 23 | boundary_high | 61505 | 13290460 | 291.60 | Gold cap 3033600 kHz |
| 24 | boundary_low | 61504 | 1410162 | 276.60 | Gold cap 3033600 kHz |
| 25 | boundary_high | 61505 | 13272634 | 287.60 | Gold cap 3033600 kHz |
| 26 | saxpy_npu65536 FAILED / excluded | 65536 | — | — | Gold cap 3033600 kHz |

The isolated adjacent transition is **61,504 → 61,505 elements**: the first large cliff identified in this bracket. Each side was then measured in three fresh processes under the same Gold cap. Larger sizes also show additional compiler cost/growth; this experiment does not claim that no other discontinuities exist. The threshold classifier was the geometric mean of measured original endpoint finalize times (3345384 us), which cleanly separates the ~1.4 s and ~13 s adjacent cases even under the cap. The exact adjacency is demonstrated by measurements, not extrapolation from a sparse graph.

## 13.7: compilation changes much more than execution

| Side | Elements | Finalize mean us (sample CoV) | Execute mean us (sample CoV) |
|---|---:|---:|---:|
| Below | 61504 | 1387064.33 (1.82%) | 276.87 (0.23%) |
| Above | 61505 | 13268441.00 (0.18%) | 289.93 (0.72%) |

One extra element changes finalization by **9.566x** (+11881376.67 us), but execution by only **1.047x** (+13.07 us). This points to a discrete compiler lowering/tiling/optimization decision or search complexity at a shape boundary, not proportional growth of the requested SAXPY arithmetic. The saved high-side diagnostics contain many ConvLayer lowering cost warnings; that supports a changed compiler path but does not prove a particular internal algorithm. The independent FP16 oracle confirms that successful outputs remain correct on both sides.

## 13.8: dynamically changing shapes

Dynamic shapes can trigger unpredictable compilation latency, memory consumption and startup stalls when crossing a compiler boundary, even when execution time changes little. Two approaches are:

1. Precompile/cache a bounded collection of expected shape graphs and reuse them. This moves compilation out of request latency but costs storage/memory, warm-up time, and a cache-miss policy for unforeseen shapes.
2. Bucket/pad requests to a small set of validated shapes, ideally avoiding an expensive compiler region. This reduces the number of graphs and makes latency predictable, but wastes padded arithmetic/bandwidth and can add batching delay. Padding and slicing must preserve semantics.

For rare or latency-critical cold shapes, a CPU/GPU fallback is also useful; that adds implementation and routing complexity. None of these strategies assumes that all possible shapes can be cached for free.

## 13.9: GPU versus NPU, finalize-only assumption

GPU best at N=512 averages **272.80 us**, process CoV 0.83%; Task 12 sustained NPU averages **131.83 us**. NPU is faster per execution. Under the requested assumption that only NPU finalization costs startup and GPU startup is zero:

`F + K*E_NPU < K*E_GPU`, hence `K > 5049.33/(272.80−131.83) = 35.819`.

The first winning integer is **K=36**. This is a deliberately incomplete startup model, corrected next.

## 13.10: full-process startup experiment and engine choice

Using only profiler fields, run each engine in fresh processes with R=1,64,512,2048 internal repetitions, three processes per R, alternating engines. Record host_wall_us (the same field shown by single-run `-v`) and native execution timing. Fit **H(R)=S+R×E** separately by ordinary least squares. Varying R separates roughly fixed cost from repeated execution rather than simply treating all host wall time as setup. NPU uses reps=R; GPU uses inner_reps=R. GPU R operations occur inside one kernel and average their result, so this model describes that provided repetition mechanism, not R independent GPU launches.

| Engine | R | Mean host wall us | Host-wall sample CoV | Native mean us/logical matmul |
|---|---:|---:|---:|---:|
| startup_npu | 1 | 437666.67 | 16.40% | 150.67 |
| startup_npu | 64 | 499666.67 | 3.10% | 129.40 |
| startup_npu | 512 | 536333.33 | 10.97% | 128.67 |
| startup_npu | 2048 | 729666.67 | 6.56% | 128.80 |
| startup_gpu | 1 | 218666.67 | 0.26% | 289.90 |
| startup_gpu | 64 | 234666.67 | 6.61% | 317.27 |
| startup_gpu | 512 | 396333.33 | 4.66% | 341.02 |
| startup_gpu | 2048 | 911333.33 | 1.93% | 346.58 |

| Fit | Fixed S (us) | Slope E (us/operation) | Residual RMSE (us) |
|---|---:|---:|---:|
| NPU | 465637.49 | 129.822 | 47191.17 |
| GPU | 217596.66 | 339.281 | 12683.37 |

Both fixed costs are substantial. They include backend/runtime/context initialization, graph compilation or OpenCL program setup, allocation, input initialization/transfer, warm-up where present, output readback/checksum, process teardown, and adb/transport overhead. They are not just NPU graphFinalize. GPU has setup too even though its event timer reports none of it. Cache state and transport noise remain; the RMSE and process CoVs show the uncertainty rather than implying microsecond-level precision in a fixed intercept.

The fitted total-time comparison is `S_NPU+K*E_NPU < S_GPU+K*E_GPU`. Therefore **K > (465637.49−217596.66)/(339.281−129.822)**, giving first winning **K=1185**. This is **1149 more operations** than the finalize-only K=36 model. If the fitted startup estimates are combined instead with the single-operation execution means from 13.9, the first winning K is **1760**; this alternative makes the sensitivity to different repetition mechanisms explicit. GPU per-operation timing increases in the long inner loop relative to R=1, so the two K models should not be silently treated as identical. These are approximate workload/transport-specific estimates, not universal hardware constants.

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
