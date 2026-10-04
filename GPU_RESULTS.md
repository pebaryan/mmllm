# GPU comparison on an ASUS AT5IONT-I (Atom D525)

Nine GPUs were run through the same mmllm engine (GLSL fragment shaders over headless EGL) on one machine:
**ASUS AT5IONT-I**, Intel Atom D525 (1.8 GHz, 2 cores / 4 threads, SSSE3, no SSE4.1), 8 GB RAM,
Zorin OS 17.3, kernel 6.8, Mesa 23.2, `nouveau` / `radeon`, plus NVIDIA's proprietary 580 driver for the
Pascal cards. The onboard GT218 is disabled whenever a card is accepted in the PCIe slot (which is only an
x1 gen-1 link on this board; irrelevant here because the weights stay in GPU memory).

All numbers are tokens/s, **best of 3 runs**, greedy decoding (`--temperature 0`), prompt `42,128,256`.
Every card produced **byte-identical token streams** (checked on TinyStories-3M against the Mac mini
reference), so these are speed comparisons of a correct engine.

## Results

Open-source drivers (nouveau, radeon):

| Card | Chip / driver | VRAM | Clocks used | 3M | 33M | GPT-2, LM head on CPU | GPT-2, LM head on GPU |
|---|---|---|---|---|---|---|---|
| **Quadro K2000** | Kepler GK107, nouveau | 2 GB GDDR5 | `pstate 0f`: 953 MHz core / 4000 MHz mem | 42.3 | **67.6** | **14.9** | **33.2** |
| GeForce 8600 GTS | Tesla G84, nouveau | 256 MB GDDR3 | boot: 675 / 1458 / 1008 MHz | **45.6** | 42.1 | 12.8 | not run (256 MB) |
| Quadro P620 | Pascal GP107, nouveau | 2 GB GDDR5 | boot clocks (cannot be raised) | 41.0 | 41.5 | 12.2 | 19.7 |
| Quadro P2000 | Pascal GP106, nouveau | 5 GB GDDR5 | boot clocks (cannot be raised) | 40.5 | 41.4 | 11.6 | 19.1 |
| Quadro P400 | Pascal GP107, nouveau | 2 GB GDDR5 | boot clocks (cannot be raised) | 37.5 | 27.7 | 10.3 | 14.0 |
| Quadro 2000 | Fermi GF106, nouveau | 1 GB GDDR5 | state 07: 405 MHz core / 324 MHz mem (cannot be raised) | 29.3 | 30.0 | 10.0 | 14.2 |
| Radeon HD 6450 | Caicos, radeon (r600g) | 1 GB DDR3 | `high` (`auto` gave the same) | 24.0 | 21.9 | 7.4 | 9.7 |
| GT218 (onboard ION) | Tesla GT218, nouveau | 512 MB DDR3 | `pstate 0f`: 535 / 1230 / 790 MHz | 36.8 | 20.2 | 8.7 | 10.9 |
| Quadro 4000 | Fermi GF100, nouveau | 2 GB GDDR5 | state 03: 50 MHz core / 135 MHz mem (cannot be raised) | 17.0 | 7.5 | 3.1 | 3.2 |

The same Pascal cards on NVIDIA's **proprietary 580.95** driver (P0 under load, e.g. 1252 MHz core / 2004 MHz
memory on the P400):

| Card | 3M | 33M | GPT-2, LM head on CPU | GPT-2, LM head on GPU | vs nouveau (33M / GPT-2 GPU head) |
|---|---|---|---|---|---|
| Quadro P400 | 26.8 | 15.8 | 7.1 | 7.9 | 0.57x / 0.56x |
| Quadro P620 | 26.7 | 15.8 | 7.0 | 7.9 | 0.38x / 0.40x |
| Quadro P2000 | 26.3 | 13.0 | 6.3 | 6.5 | 0.31x / 0.34x |

For reference, the Mac mini (GeForce 9400M, Core 2 Duo) measured 84 (3M), 15.5 (33M) and 9.1 (GPT-2).

Needle 3 on the **CPU** decodes at about 10 tokens/s on this machine with every card installed (it does
not use the GPU by default). Its opt-in `--gpu` path is correct on the GT218 and K2000
(`--gpu-check` passes) but slower than the CPU (3.0 and 3.5 tokens/s), because it pays a draw plus a
readback for each of ~110 matrix-vector products per token and those costs are set by the Atom and the
driver, not the GPU. On the HD 6450 `--gpu-check` fails (relative error 0.11): the shader decodes packed
2-bit weights from an integer texture, which the r600 driver does not handle correctly.

## What the numbers say

* **Best overall: Quadro K2000 at `pstate 0f`.** GPT-2 small reaches 33 tokens/s, over three times the
  onboard GT218. This workload is memory-bandwidth bound where the clocks can be changed: reclocking the
  K2000's memory from 648 to 4000 MHz took GPT-2 (GPU head) from 12.6 to 33.2 and TinyStories-33M from
  26.1 to 67.6.
* **Tiny models are limited by the CPU, not the GPU.** On TinyStories-3M the 8600 GTS (45.6) beats
  every faster card; the cost is per-draw driver work on the Atom.
* **On this slow CPU the lighter driver wins.** The P2000 is 1.5x (3M), 3.2x (33M) and 2.9x (GPT-2, GPU head)
  faster on nouveau than on NVIDIA's proprietary driver, and the three Pascal cards are nearly
  indistinguishable from each other under the proprietary driver (the bigger P2000 is even a little slower).
  `time` on the 33M run under 580 shows user 3.95 s + sys 2.85 s of 6.8 s wall: the process is CPU-bound by
  the proprietary driver's userspace and system-call overhead. `__GL_THREADED_OPTIMIZATIONS` and `__GL_YIELD`
  make no difference (threaded optimizations on is worse).
* **Among the Pascal cards on nouveau, size does not help much.** The 2 GB P620 (512 cores, 128-bit) matches the
  5 GB P2000 (1024 cores, 160-bit) and beats the 64-bit P400 on the larger models; they run at their boot
  clocks, since reclocking Pascal needs PMU firmware that nouveau does not have (`pmu: firmware unavailable`).
* **The Fermi cards are held back by nouveau, not by silicon.** On kernel 6.8 writing `0f` to
  `/sys/kernel/debug/dri/0/pstate` returns "Function not implemented" for both Fermi cards, so they stay at
  their boot clocks (the Quadro 4000 boots at 135 MHz memory). Kepler and the Tesla-family cards can
  be reclocked. NVIDIA's legacy 390 driver supports Fermi but its stock Ubuntu source does not build on
  6.8 (`get_user_pages_remote` signature), so it would need an older kernel or patches.
* **GPU memory matters for GPT-2.** The default 200 MB budget puts GPT-2's LM head on the CPU. Cards
  with at least 1 GB can keep it on the GPU: `MMLLM_GPU_BUDGET_MB=800 MMLLM_LM=gpu`. Do not do this on a
  256 MB card; spilling to system memory made lookups several times slower on the Mac mini and
  eventually failed with `ENOMEM`.

## Method and caveats

* `tools/gpu_bench/bench.sh` waits until the 1-minute load average is below 0.25, runs each case three
  times and keeps the best. `SKIP_GPULM=1` skips the GPU-head case for small-VRAM cards.
* The desktop was slimmed first (indexer, software centre, PackageKit, Evolution, Bluetooth, printing and
  other background services disabled): leaving it running cost roughly 10-50% on the CPU-bound cases.
* **The GT218 was measured before that cleanup**, with a busy desktop and single runs, so its numbers are
  pessimistic relative to the others. It was not re-measured.
* The 8600 GTS had no monitor detected during its run (its DVI outputs report `disconnected`); the
  other cards had one attached or were headless with GDM running. The Pascal cards ran with no monitor
  attached.
* HD 6450: forcing `power_dpm_force_performance_level=high` changed results by 0-3% versus `auto`.
* Cards are compared at the clocks nouveau/radeon allowed, not at their rated maximums. The P620 and
  P2000 results on nouveau are at whatever clocks the VBIOS left them.
* A **P106-100** (Pascal mining card with no display output) was never detected: the board's slot seems to
  switch its single lane between the slot, the onboard JMicron SATA controller and the onboard GT218 using
  card-presence detection, and with the P106-100 installed the slot stayed on the SATA controller and the GT218
  stayed enabled. Cards with display outputs (the P400, P620, P2000) were accepted. This is an inference from
  the PCIe topology (a PLX PEX8608 switch with seven downstream ports, only one ever linking), not verified.

## Using NVIDIA's proprietary driver

The proprietary 580 driver supports Pascal and works with the engine, which prefers NVIDIA's EGL device
whenever the NVIDIA kernel driver is loaded (`MMLLM_EGL=surfaceless` skips this; `MMLLM_EGL_DEVICE=<n>`
selects a device explicitly). Without that, Mesa's surfaceless platform would silently give the llvmpipe
software renderer. It is slower on this Atom (see above), so it is not recommended here. Notes if you try it:

* The DKMS build took about 50 minutes on the Atom because it built for every installed kernel; remove
  headers of old kernels first to cut it.
* `nvidia-kernel-common-580` ships the files that blacklist nouveau. If other packages depend on it, move the
  files aside with `dpkg-divert --local --rename --add` instead of purging it, then `update-initramfs -u`.
* While 580 is installed, every card older than Maxwell (everything except the Pascal cards here) has no driver.

## Gotchas found

* If nobody is logged in on the console, an ordinary user gets `Permission denied` on
  `/dev/dri/renderD128` and Mesa **silently falls back to llvmpipe** (software, ~3 tokens/s, correct
  output). Add the user to the `render` and `video` groups, and confirm the renderer with
  `./build/mmllm --self-test | grep -o "(NVC3\|(NVE7\|(NV13[67]\|(AMD\|(llvmpipe\|NVIDIA 580"`.
* The default of `mmllm` is sampling at temperature 1.0 with a random seed. Always pass
  `--temperature 0` when comparing machines.
* Leftover NVIDIA 580 driver packages blacklist nouveau and cannot drive pre-Maxwell cards: the GT218
  had no driver until they were removed.
* `pstate` writes need root and debugfs; they do not persist across reboots.
* A background watcher must not use `kill -0 <pid>` on a root-owned process (it fails with EPERM and looks
  like "exited"); use `ps -p <pid>`.
