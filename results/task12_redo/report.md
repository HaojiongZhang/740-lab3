# Task 12: The Boundary and the Copy

Instructions: Lab3_F26 (1).pdf, Task 12.1–12.7, pages 21–22.

## Method and verified setup

Used the connected SM8850P board, lab3 container, and the existing QAIRT 2.40.0.251030 provisioning. No QNN/reference client or profiler source was edited. Implemented only kernels/lab3/memshare/memshare.cpp, switched MEMSHARE_STUB_ONLY to 0, and built/pushed it with `bash scripts/build_student.sh memshare`. The library resolves rpcmem through librpcmem.so or libcdsprpc.so, requires alloc/free/to_fd, initializes once, allocates page-rounded cached buffers (default heap -1, flags 1), returns their real file descriptor, and releases them through rpcmem. Invalid/overflowing sizes fail; failed rpcmem allocations cannot fall through to a mismatched heap/free pair. Initialization/deinitialization are paired when provided by the runtime.

Each of the three N values has four cases: ordinary fp16 sustained mode (12.1), ordinary fp16 default mode (12.2), spike client baseline, and spike client with the student shared allocator (12.4). Each case has three fresh process dispatches, each with 16 internal graphExecute timings after an untimed warm-up. The 36 process runs therefore contain 576 timed graph executions. Cases alternate within each repetition to reduce ordering bias; runs are serial, with 3 s between cases and 2 s between sizes. Timing is execute_us_mean, excluding graph construction, finalization, memory registration, warm-up, and allocation. Raw output and every parsed field are in raw.jsonl; measurements.csv stores per-process results. The script's three fresh dispatches implement the same repetition structure as the handout's --repeat 3.

Two variability measures must be distinguished: **process CoV** below is the sample CoV of the three process execution means. **Internal CoV** is the client's population CoV over its 16 timed executions; the table gives the mean of those three reported CoVs. GFLOP/s is averaged over the three process results. No clocks or power modes were changed outside the explicitly selected client perf_mode. The ordinary and spike client comparison is kept within each client for speedups; Part B does not assume the two binaries have identical overhead.

## 12.1: sustained-mode size scaling

| N | Execute mean (us) | Process CoV | GFLOP/s mean | Finalize mean (us) | Finalize CoV | Internal CoV mean |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 74.20 | 0.54% | 452.22 | 4427.00 | 1.30% | 4.742% |
| 512 | 131.83 | 1.31% | 2036.41 | 5049.33 | 2.97% | 6.089% |
| 1024 | 397.47 | 0.43% | 5402.99 | 9566.33 | 3.05% | 1.818% |

Execution time grows **1.777x** from 256→512 and **3.015x** from 512→1024. Doubling N multiplies arithmetic (2N³ FLOPs) by 8 and tensor bytes (three FP16 N×N tensors) by 4. Both observed ratios are far below 8, and also below 4. Thus runtime over this range is not explained by a fixed-throughput arithmetic-only model. A fixed boundary/dispatch component matters strongly at N=256, while size-dependent data movement and computation become more important as N grows. The larger ratio at the second doubling is consistent with that mixture. The timings alone do not uniquely separate copies, hardware compute, bandwidth, and utilization; Part B directly tests the copy contribution.

## 12.2: default performance mode

| N | Execute mean (us) | Process CoV | GFLOP/s mean | Finalize mean (us) | Internal CoV mean | Default/sustained time |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 737.57 | 13.85% | 46.08 | 6770.67 | 16.656% | 9.94x |
| 512 | 982.50 | 0.51% | 273.22 | 7582.33 | 4.734% | 7.45x |
| 1024 | 1724.07 | 4.31% | 1247.17 | 12094.67 | 14.847% | 4.34x |

Default-mode growth is **1.332x** then **1.755x**. Default is slower at all sizes, with much greater internal variation at N=1024. The measured internal CoVs and process CoVs describe different sources of variation and are both retained; full per-process values are in measurements.csv. The supplied sustained path configures performance/DCVS and RPC polling behavior, whereas default leaves the runtime's default behavior. Changes in clocks, sleep/wake behavior, and transport/polling overhead can therefore change the observed growth curve. The 12.1 conclusion is conditional on sustained mode and these sizes; a scaling curve alone is insufficient to label the NPU's arithmetic or memory bottleneck independently of its operating state. In particular the larger default internal spread should not be mistaken for changing numerical results: fingerprints remain identical.

## 12.3: finalization versus execution at N=512

In sustained mode, finalization averages **5049.33 us** versus **131.83 us** per execute, so compilation costs approximately **38.30 executions**. Default mode gives 7582.33 us finalization and 982.50 us execution; changing execution speed changes the relative burden of compilation too.

Count finalize when comparing cold/startup latency, a one-off graph, or a shape/graph that needs recompilation. For K executions reusing the same finalized graph, include F/K per execution (or total F+K×E). It becomes negligible only when amortized over sufficiently many executions, or when a legitimately reusable precompiled/finalized graph moves this work outside the relevant latency budget. Graph setup, backend loading, allocations and registration may add other one-time costs. The benchmark's execute mean excludes those costs; it is not application startup latency.

## Part B implementation and correctness

The shared variant dynamically loads libmemshare.so; all nine shared process runs explicitly report **shared_impl=memshare**. The nine corresponding baseline runs report **none_clientbuf**, not builtin_rpcmem. Each shared run successfully registers the three tensor memory handles and completes all graph executions. For every N, both fingerprint_sum and fingerprint_frob exactly match the baseline in every repetition. These two fingerprints are the harness's correctness criterion; they are not an exhaustive element-by-element numerical oracle.

An additional ARM64 board-side test loaded the actual library and passed 21 simultaneous/repeated allocations at sizes 1, 4095, 4096, 4097, 393216, 1572864 and 6291456 bytes. It checked page alignment, readable/writable memory, valid and stable exported FDs, FD closure after rpcmem_free, NULL handling, rejection of zero/oversized/overflowing requests, and rejection of an ordinary malloc buffer as shareable. The exact PASS output is in allocator_check.txt; the test source is scripts/check_task12_memshare.c. A shared allocation remains live until its QNN memory registrations are deregistered by the supplied client.

## 12.4: shared versus baseline execution

| N | Baseline mean us (process CoV) | Shared mean us (process CoV) | Speedup | Time saved us | Three-tensor payload |
|---|---:|---:|---:|---:|---:|
| 256 | 73.17 (0.42%) | 72.67 (3.51%) | 1.007x | 0.50 | 393,216 bytes |
| 512 | 132.73 (1.07%) | 122.03 (0.62%) | 1.088x | 10.70 | 1,572,864 bytes |
| 1024 | 394.90 (0.13%) | 265.60 (0.08%) | 1.487x | 129.30 | 6,291,456 bytes |

Fingerprint evidence (identical for baseline and shared, all repetitions):

| N | Sum | Frobenius norm |
|---|---:|---:|
| 256 | 4196458.5625 | 16421.895675 |
| 512 | 33520107.875 | 65522.253453 |
| 1024 | 268535101.5 | 262342.661784 |

The benefit grows from nearly no change at 256 to **1.487x** at 1024. The allocation holds A, B and C: 3×N²×2 bytes, which grows fourfold at each doubling. Removing the client-buffer boundary copies therefore saves more absolute work at larger N. At 256 the entire useful payload is only 393,216 bytes, and fixed execution/dispatch plus shared-memory handling dominate the tiny copy saving. The small measured 256 benefit is comparable to normal per-run variation; it should be described as approximately equal performance, not a robust large optimization. Speedup need not follow an exact N² law because computation, fixed overhead, caching, and coherence costs remain.

## 12.5: why rpcmem rather than malloc

rpcmem allocates memory supported by the FastRPC shared-memory path and provides an exportable DMA-BUF/ION-style file descriptor with a managed backing allocation. QNN can register and map that backing memory into the accelerator/DSP address space. A normal malloc pointer only identifies a region in the CPU process's virtual address space; it supplies neither an exportable backing-object FD nor the registration/lifetime/coherence contract required by this shared path. The NPU cannot use that CPU virtual pointer directly, so a raw client buffer requires copying or another supported registration mechanism.

This does **not** require physically contiguous pages in all cases. The installed SDK's rpcmem.h describes RPCMEM_HEAP_ID_SYSTEM as non-contiguous system physical memory intended for cDSP/aDSP subsystems with an SMMU. The important property is exportable, mappable backing memory and its lifetime, not physical contiguity of an ordinary virtual range. The implementation uses the starter's requested default heap (-1), cached flags (1), and verifies that rpcmem_to_fd actually returns a valid FD.

## 12.6: one allocation for three tensors

The supplied client makes one allocation and one FD, with A at offset 0, B at `bytes`, and C at `2*bytes`. It registers three distinct QNN handles using that FD and the corresponding offsets. The included QnnHtpMem.h explicitly defines shared-buffer FD+offset registration and distinct handles per offset. One allocation reduces allocation/FD/backing-buffer management and offers a simple common lifetime; it does not eliminate the three tensor registrations. The requested sizes yield page-aligned tensor offsets.

Separate allocations are preferable when tensors have different lifetimes, ownership, producers/consumers, memory/cache policies, or need to be replaced/reused independently. For example, long-lived weights can remain registered while input/output buffers rotate through independent pipeline slots. Separate blocks can also make pooling and independent release easier, instead of retaining one combined allocation until the longest-lived tensor is finished.

## 12.7: data still moves with shared memory

At N=1024 the shared execution is 265.60 us versus 394.90 us baseline: removing the copy saves about 32.7% of baseline execution time, leaving 67.3%. Shared host backing memory is still external memory from the accelerator's perspective. The NPU must fetch input/weight data through its memory hierarchy into on-chip execution storage and write output data back to shared memory. Sharing removes redundant CPU/client-buffer boundary copying, not these required reads/writes or matrix arithmetic. Cache clean/invalidate/coherence work, layout transfers/repacking where used by the backend, and dispatch/synchronization can remain as well. The observations do not quantify each backend-internal category separately.

Allocating/filling inputs and registering buffers occur before the timed rep loop; they matter for full application cost but do not explain the measured per-execute time savings. Finalization is also outside this loop. Thus “zero-copy” describes the removed boundary copy, not zero memory traffic or zero setup.

## Reproduction and files

Inside /workspace in the container:

```bash
bash scripts/build_student.sh memshare
PYTHONPATH=/workspace python scripts/measure_task12.py --output results/task12_redo --pause 3 --max-temp 60
python profiler.py matmul npu -p size=512 --repeat 3
python profiler.py matmul npu -p size=512 -p perf_mode=default --repeat 3
python profiler.py matmul npu --variant spike -p size=1024
python profiler.py matmul npu --variant spike -p size=1024 -p memshare=./libmemshare.so
```

The runner includes 256, 512, and 1024 for every case, explicitly uses reps=16, and saves each result immediately. Use the same host adb setup as the earlier tasks; these measurements used ANDROID_ADB_SERVER_ADDRESS=192.168.65.254 for the currently verified Docker host address. Results/interpretation can be regenerated with `python scripts/report_task12.py --output results/task12_redo` on the laptop using only the standard library.

Only the Task 12 student allocator and new measurement/report/test files were added or edited for this task. Task 10 and Task 11 code and their deployed DSP library were preserved. No SDK, QNN, reference client, or profiler logic was edited for Task 12.

## Fresh rerun temperature verification

All 36 runs logged relevant die-sensor temperatures. Post-dispatch hottest-sensor readings ranged from 30.8 to 47.3 °C. The runner checked temperatures before each dispatch and stopped on unreadable telemetry or readings at/above 60 °C. Three-second pauses were used between cases. The prior Task 12 dataset was preserved separately.
