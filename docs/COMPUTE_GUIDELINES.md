# Compute Environment & Optimization Guidelines
**Machine:** Mac Mini (Apple Silicon)  
**Date:** 2026-03-19 (updated)  
**Author:** Aaron Choi Ramos  
**Replaces:** COMPUTE_GUIDELINES.md dated 2026-03-16  

---

## 1. Hardware Profile

| Resource | Specification |
|----------|---------------|
| CPU | Apple Silicon — 4 Performance cores + 6 Efficiency cores (10 total) |
| macOS Scheduler | QoS-aware — routes sustained compute to P-cores, background tasks to E-cores |
| Python GIL | Global Interpreter Lock blocks true CPU parallelism within a single process |
| NumPy/SciPy | Uses Apple Accelerate framework — already optimized for single-thread FFT/BLAS |

**Key constraint:** Each single-threaded Python process can saturate exactly one core. Our RK4 + spectral Laplacian simulations are CPU-bound and single-threaded, meaning one simulation = one P-core maxed at 100%.

---

## 2. Parallelization Strategy

### 2.1 When to Parallelize

Parallelize at the **study level** (multiple independent simulations), NOT at the **simulation level** (inner RK4 loops). Each individual evolution is a tightly coupled sequential computation that doesn't benefit from naive parallelism. The wins come from running multiple configurations simultaneously.

| Scenario | Parallelize? | Method |
|----------|-------------|--------|
| Parameter sweeps (multiple configs) | **YES** | `multiprocessing.Pool(workers=4)` |
| Single long evolution (T=500) | No | Let it run on one P-core |
| Post-processing / data aggregation | Usually no | Fast enough single-threaded |
| Numba-accelerated inner loops | **Maybe** | `@njit(parallel=True)` with `prange` — test first |

### 2.2 Target Worker Count

**Use 4 workers** for parameter sweeps. This saturates all P-cores without spilling onto E-cores, which would give slower per-job performance. macOS will naturally route these to the P-cores.

```python
import multiprocessing as mp

N_WORKERS = 4  # Match P-core count

with mp.Pool(N_WORKERS) as pool:
    results = pool.map(run_single_config, config_list)
```

If a study has very many short jobs (e.g., 100+ configs at T=50), you can experiment with `N_WORKERS = 6` to also use some E-cores, but expect ~30% less throughput per job on E-cores. For long jobs (T=500), stick with 4.

### 2.3 Required Boilerplate

macOS requires the `__main__` guard for multiprocessing. **Every script that uses multiprocessing MUST include this:**

```python
if __name__ == '__main__':
    main()
```

Without this, macOS will crash or hang on `Pool` creation due to the `spawn` start method used on macOS (not `fork`).

---

## 3. Incremental Checkpointing (MANDATORY)

**Every simulation that takes longer than 5 minutes MUST use incremental checkpointing.** We have lost data multiple times from crashes, session timeouts, and killed processes. Writing results only at the end is not acceptable.

### 3.1 Checkpoint Architecture

The checkpoint system lives in `engine/checkpoint.py` and provides a `CheckpointManager` class. Study scripts should NEVER implement their own checkpoint logic.

**Checkpoint flow:**
1. Before starting, check if a completed output file exists (skip if `completed: true`)
2. If a `.checkpoint.json` file exists, resume from it
3. During evolution, save checkpoint every `checkpoint_interval` time units (default: 50.0)
4. Checkpoints use atomic writes: write to `.tmp`, then `os.replace()` to final path
5. On completion, write final output JSON with `completed: true` and delete checkpoint file
6. Final output must NOT contain phi/phi_dot field arrays (too large, not needed)

### 3.2 Atomic Writes

**NEVER write directly to the output path.** Always write to a temp file first, then atomically rename:

```python
import os, json

def atomic_write_json(data, filepath):
    """Write JSON atomically to prevent corruption on crash."""
    tmp_path = str(filepath) + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, str(filepath))  # Atomic on POSIX
```

### 3.3 Checkpoint File Naming

Checkpoint files sit next to the output file with a `.checkpoint.json` suffix:

```
outputs/phase2/cube_polarized.json              # Final output
outputs/phase2/cube_polarized.json.checkpoint    # In-progress checkpoint (deleted on completion)
```

### 3.4 Checkpoint Content

Checkpoints contain everything needed to resume:

```json
{
  "completed": false,
  "current_step": 4000,
  "current_time": 200.0,
  "time_series": {
    "times": [0, 0.5, 1.0, ...],
    "E_total": [...],
    "max_amplitude": [...]
  },
  "field_state": {
    "phi_b64": "<base64-encoded numpy array>",
    "phi_dot_b64": "<base64-encoded numpy array>",
    "shape": [64, 64, 64]
  },
  "config": { ... }
}
```

**IMPORTANT:** The `field_state` section is large (~4 MB for N=64, ~32 MB for N=128). It MUST be stripped from the final output JSON. Only checkpoints carry the field arrays.

### 3.5 Cache Detection

When a script starts, it checks the output path:
- If final JSON exists AND has `"completed": true` → print `CACHED: [config name]` and return immediately
- If checkpoint exists → print `RESUMING: [config name] from T=XX.X` and continue from saved state
- Otherwise → start fresh

**CRITICAL for convergence runs:** The cache key MUST include N_grid. A cached N=64 result must NOT satisfy an N=128 request. Include N_grid in the output filename or in the JSON metadata check:

```python
def is_valid_cache(filepath, required_N_grid):
    """Check if cached result matches the requested resolution."""
    if not os.path.exists(filepath):
        return False
    with open(filepath) as f:
        data = json.load(f)
    if not data.get('completed', False):
        return False
    stored_N = data.get('parameters', {}).get('N_grid', None)
    if stored_N != required_N_grid:
        return False
    return True
```

### 3.6 Using the Checkpoint System

All study scripts use the same pattern:

```python
from engine.checkpoint import run_with_checkpointing

def run_single_config(config):
    from engine.evolver import SexticEvolver
    
    evolver = SexticEvolver(**config['params'])
    evolver.initialize(config['initial_conditions'])
    return run_with_checkpointing(evolver, config, config['output_path'])
```

No checkpoint logic in study scripts. Ever.

---

## 4. Process Persistence & Timeout Prevention

### 4.1 The Problem

Claude Code sessions can time out, killing child processes. macOS can also deprioritize background processes. Long simulations (T=500 at N=128 takes ~80 min) are especially vulnerable.

### 4.2 Detaching Long Runs with nohup

For any run expected to take longer than 30 minutes, launch with `nohup` to detach from the terminal session:

```bash
cd ~/Desktop/swirl-theory/geometric_binding_study
nohup python studies/my_study.py > logs/my_study.log 2>&1 &
echo $! > logs/my_study.pid
echo "Launched PID $(cat logs/my_study.pid)"
```

This ensures:
- The process survives Claude Code session timeouts
- stdout/stderr are captured in a log file
- The PID is saved for monitoring

### 4.3 Monitoring Detached Runs

```bash
# Check if still running
kill -0 $(cat logs/my_study.pid) 2>/dev/null && echo "RUNNING" || echo "FINISHED"

# Watch the log
tail -f logs/my_study.log

# Check CPU usage
ps -p $(cat logs/my_study.pid) -o %cpu,rss,etime
```

### 4.4 Preventing macOS from Deprioritizing

macOS may throttle background Python processes. Two mitigations:

```bash
# Option 1: Keep terminal in foreground (simplest)
# Just don't close the terminal window

# Option 2: Set QoS to user-initiated (higher priority)
taskpolicy -b off python studies/my_study.py
```

### 4.5 Handling Orphan Processes

When a parent process is killed, `multiprocessing.Pool` workers become orphans consuming CPU. Always check for these:

```bash
# Find orphan Python processes
ps aux | grep python | grep -v grep

# Kill all Python processes (CAREFUL — only when you know what's running)
pkill -f "python studies/"
```

### 4.6 Graceful Shutdown

Study scripts should handle SIGINT/SIGTERM to save a final checkpoint before dying:

```python
import signal

def graceful_shutdown(signum, frame):
    print("Caught signal, saving checkpoint...")
    # checkpoint_manager.save() is called here
    raise SystemExit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
```

---

## 5. Disk Space & Temp File Management

### 5.1 The Problem

Claude Code writes working files to a temp directory. Long-running or repeated simulations accumulate large amounts of data that is never cleaned up. Checkpoint files with embedded field arrays are especially large.

### 5.2 Storage Budget

| File Type | Size (N=64) | Size (N=128) |
|-----------|-------------|--------------|
| Final output JSON (no fields) | ~50-200 KB | ~50-200 KB |
| Checkpoint with field arrays | ~4 MB | ~32 MB |
| Full time series (10k steps) | ~2 MB | ~2 MB |

For a 20-config study at N=64: ~100 MB in checkpoints during the run, ~4 MB in final outputs.

### 5.3 Cleanup Rules

**Mandatory cleanup after each study completes:**

1. Delete all `.checkpoint.json` files for completed configs
2. Delete all `.tmp` files (failed atomic writes)
3. Verify all final outputs have `completed: true`
4. Print disk usage summary

```python
import glob, os

def cleanup_study(output_dir):
    """Remove checkpoint and temp files after study completes."""
    checkpoints = glob.glob(os.path.join(output_dir, '*.checkpoint*'))
    temps = glob.glob(os.path.join(output_dir, '*.tmp'))
    
    total_freed = 0
    for f in checkpoints + temps:
        size = os.path.getsize(f)
        os.remove(f)
        total_freed += size
    
    print(f"Cleanup: removed {len(checkpoints)} checkpoints, "
          f"{len(temps)} temp files, freed {total_freed / 1e6:.1f} MB")
```

### 5.4 Field Array Policy

- **Checkpoint files:** INCLUDE phi and phi_dot (needed for resume)
- **Final output JSON:** NEVER include phi or phi_dot (too large, not needed for analysis)
- **If you need field snapshots for visualization:** Save as separate `.npz` files in a `fields/` subdirectory, not inside the JSON

### 5.5 Claude Code Temp Directory

Claude Code may write files to its own temp workspace (often `/tmp/` or `~/.claude/`). These are NOT in the project directory and can accumulate. Periodically check:

```bash
# Check Claude Code temp usage
du -sh /tmp/claude* 2>/dev/null
du -sh ~/.claude/ 2>/dev/null

# Clear if needed (only when no runs are active)
rm -rf /tmp/claude_workspace_* 2>/dev/null
```

### 5.6 Output Directory Structure

Keep outputs organized. Don't dump everything in one flat directory:

```
outputs/
  phase1/           # Baseline data
  phase2/           # Main geometry study data
  convergence/      # N=128 convergence tests
  icosahedron_threshold/  # ce_15 variant tests
  figures/          # Generated figures
  logs/             # nohup logs and PIDs
```

Each subdirectory should be self-contained. Old studies should be archived (compress to .tar.gz) if no longer actively used.

---

## 6. Memory Considerations

### 6.1 Per-Simulation Memory

Each 64-cubed grid simulation uses approximately:

| Array | Size | Count | Total |
|-------|------|-------|-------|
| phi field (float64) | 64^3 x 8 bytes = 2 MB | 1 | 2 MB |
| phi_dot field | 2 MB | 1 | 2 MB |
| RK4 intermediates (k1-k4) | 2 MB each | ~8 | 16 MB |
| FFT workspace | ~4 MB | 1 | 4 MB |
| Diagnostics buffer | ~1 MB | 1 | 1 MB |

**Estimate: ~25 MB per simulation**

With 4 workers: ~100 MB total. Memory is NOT a bottleneck for N=64 grids.

### 6.2 Higher Resolution Caution

If we move to N=128 grids, memory scales as N^3:

| Grid | Per-Sim Memory | 4 Workers |
|------|---------------|-----------|
| 64^3 | ~25 MB | ~100 MB |
| 128^3 | ~200 MB | ~800 MB |
| 256^3 | ~1.6 GB | ~6.4 GB |

At N=256, check available RAM before running 4 workers.

---

## 7. Numba Optimization

### 7.1 Current Usage

The existing codebase uses Numba's `@njit` for hot loops in `fast_ops.py`. This is effective for single-thread acceleration.

### 7.2 Parallel Numba

For inner loops that iterate over grid points independently (e.g., potential evaluation, energy density computation), Numba's `parallel=True` with `prange` can distribute work across cores:

```python
from numba import njit, prange

@njit(parallel=True)
def compute_energy_density(phi, phi_dot, grad_phi_sq, m2, g4, g6):
    N = phi.shape[0]
    H = np.empty_like(phi)
    for i in prange(N):
        for j in range(N):
            for k in range(N):
                p = phi[i, j, k]
                H[i, j, k] = (0.5 * phi_dot[i, j, k]**2 
                              + 0.5 * grad_phi_sq[i, j, k]
                              + 0.5 * m2 * p**2
                              - (g4/24) * p**4
                              + (g6/720) * p**6)
    return H
```

**IMPORTANT:** Only use `parallel=True` for embarrassingly parallel loops. Do NOT use it for the RK4 stepper itself — the time steps are sequential by nature.

**CAUTION:** When combining `multiprocessing.Pool` with Numba `parallel=True`, the total thread count can explode (4 processes x N Numba threads). To prevent oversubscription:

```python
import os
os.environ['NUMBA_NUM_THREADS'] = '1'  # Set BEFORE importing numba

# OR if using Pool + Numba, limit Numba threads per worker:
os.environ['NUMBA_NUM_THREADS'] = str(max(1, 4 // N_WORKERS))
```

**General rule:** If using `multiprocessing.Pool`, set `NUMBA_NUM_THREADS=1`. If running a single simulation, let Numba use up to 4 threads.

---

## 8. Practical Speedup Estimates

### 8.1 Current Baseline (Sequential)

For a sweep of K configurations, each taking time T_single:

**Sequential time: K x T_single**

### 8.2 With 4-Worker Pool

**Parallel time: ~(K / 4) x T_single + overhead**

Overhead is minimal (<1 second for process creation). For our typical studies:

| Study | Configs | T_single (est.) | Sequential | 4-Worker Parallel |
|-------|---------|-----------------|------------|-------------------|
| Phase 1 baseline | 2 | ~15 min (T=500) | 30 min | 15 min |
| Phase 2 cube | 4 | ~15 min | 60 min | 15 min |
| Phase 2 icosahedron | 15 | ~15 min | 3.75 hrs | ~1 hr |
| Full sweep (316 configs, T=50) | 316 | ~1 min | 5.3 hrs | ~1.3 hrs |
| Convergence (N=128) | 4 | ~80 min | 5.3 hrs | ~80 min |

### 8.3 Amdahl's Law Reminder

Speedup is limited by the sequential fraction. If 5% of runtime is non-parallelizable (setup, I/O, merging), maximum speedup with 4 cores is ~3.5x, not 4x.

---

## 9. Monitoring During Runs

### 9.1 Activity Monitor

- **Window > CPU History** — confirm 4 P-cores are loaded during sweeps
- Each Python worker should show ~100% CPU
- E-cores should be mostly idle during heavy compute

### 9.2 Terminal Monitoring

```bash
# Quick CPU check
top -l 1 | grep "CPU usage"

# Watch Python processes specifically
watch -n 2 'ps aux | grep python | grep -v grep'

# Check nohup run status
tail -5 logs/my_study.log

# Detailed power/thermal (useful for long runs)
sudo powermetrics --samplers cpu_power -i 5000
```

### 9.3 Thermal Throttling

On long runs (T=500, multiple configs), the Mac Mini may thermally throttle if all P-cores are at 100% for extended periods. Signs: per-core clock drops, runtime increases. If this happens, consider `N_WORKERS = 3` to leave headroom.

---

## 10. Code Template

Standard pattern for parallelized study scripts with checkpointing and persistence:

```python
#!/usr/bin/env python3
"""
Study: [Name]
Parallelized across 4 P-cores with incremental checkpointing.
"""

import multiprocessing as mp
import json
import os
import signal
import sys
from pathlib import Path

# Prevent Numba thread oversubscription when using multiprocessing
os.environ['NUMBA_NUM_THREADS'] = '1'

N_WORKERS = 4  # Mac Mini P-core count


def run_single_config(config: dict) -> dict:
    """Run one simulation with checkpointing. Must be picklable (top-level function)."""
    from engine.evolver import SexticEvolver
    from engine.checkpoint import run_with_checkpointing
    
    outpath = config['output_path']
    
    # Check cache (including N_grid match)
    if os.path.exists(outpath):
        with open(outpath) as f:
            cached = json.load(f)
        if (cached.get('completed', False) and 
            cached.get('parameters', {}).get('N_grid') == config['params'].get('N_grid', 64)):
            print(f"CACHED: {config['name']}")
            return {'name': config['name'], 'file': outpath, 'summary': cached.get('final_state', {})}
    
    evolver = SexticEvolver(**config['params'])
    evolver.initialize(config['initial_conditions'])
    results = run_with_checkpointing(evolver, config, outpath)
    
    return {'name': config['name'], 'file': outpath, 'summary': results.get('final_state', {})}


def main():
    # Build config list
    configs = build_configs()  # Returns list of config dicts
    
    # Create output directory
    outdir = Path('outputs') / 'study_name'
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run parallel sweep
    print(f"Running {len(configs)} configs on {N_WORKERS} workers...")
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single_config, configs)
    
    # Merge and save summary
    summary = {r['name']: r['summary'] for r in results}
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Cleanup checkpoint and temp files
    cleanup_study(outdir)
    
    print(f"Complete. Results in {outdir}/")


def cleanup_study(output_dir):
    """Remove checkpoint and temp files after study completes."""
    import glob
    checkpoints = glob.glob(os.path.join(str(output_dir), '*.checkpoint*'))
    temps = glob.glob(os.path.join(str(output_dir), '*.tmp'))
    
    total_freed = 0
    for f in checkpoints + temps:
        size = os.path.getsize(f)
        os.remove(f)
        total_freed += size
    
    if checkpoints or temps:
        print(f"Cleanup: removed {len(checkpoints)} checkpoints, "
              f"{len(temps)} temp files, freed {total_freed / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
```

---

## 11. Checklist for Code Prompts

When writing prompts for Claude Code to implement study scripts, include:

**Parallelization:**
- [ ] Use `multiprocessing.Pool(4)` for the config sweep
- [ ] Set `NUMBA_NUM_THREADS=1` before any numba import
- [ ] Include `if __name__ == '__main__':` guard
- [ ] Import heavy modules inside worker functions (not at top level) to reduce spawn overhead

**Checkpointing & Persistence:**
- [ ] Use `run_with_checkpointing()` from `engine/checkpoint.py` — no custom checkpoint logic
- [ ] Atomic writes only (`.tmp` then `os.replace()`)
- [ ] Cache check must validate N_grid matches (not just `completed: true`)
- [ ] Strip phi/phi_dot field arrays from final JSON output
- [ ] Call `cleanup_study()` after all configs complete
- [ ] For runs > 30 min: launch with `nohup` and save PID to `logs/`

**Output & Monitoring:**
- [ ] Each worker saves results to a separate JSON file
- [ ] Print progress: config name + ETA after each completion
- [ ] Track wall-clock time for the full sweep
- [ ] ASCII-only console output (no unicode box-drawing chars)

**Data Hygiene:**
- [ ] Delete checkpoint files after successful completion
- [ ] Delete `.tmp` files after successful completion
- [ ] Never store field arrays in final output JSON
- [ ] Include full config metadata in every output file (parameters, phases, geometry)

---

*This document should be referenced whenever writing new simulation scripts for the geometric binding study.*
