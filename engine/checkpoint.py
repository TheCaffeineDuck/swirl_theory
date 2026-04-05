"""
engine/checkpoint.py
====================
Standalone checkpoint management for long-running simulations.

Provides:
  - CheckpointManager: cache detection, resume, incremental saves, finalization
  - run_with_checkpointing(): high-level wrapper for evolver.evolve()
  - cleanup_study(): post-study temp file removal

All atomic writes use .tmp -> os.replace() to prevent corruption on crash.
"""

import glob
import json
import os
import sys
import time
from pathlib import Path


def atomic_write_json(data, filepath):
    """Write JSON atomically to prevent corruption on crash."""
    tmp_path = str(filepath) + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, str(filepath))


class CheckpointManager:
    """Manages caching, resume, incremental checkpoints, and finalization."""

    def __init__(self, output_path, checkpoint_interval=50.0):
        """
        Args:
            output_path: path for final output JSON
            checkpoint_interval: save checkpoint every this many time units
        """
        self.output_path = str(output_path)
        self.checkpoint_path = self.output_path + '.checkpoint.json'
        self.checkpoint_interval = checkpoint_interval
        self._last_checkpoint_time = 0.0

    def check_cache(self, required_N_grid=None):
        """Return cached results if output exists with completed=True AND matching N_grid.

        Returns:
            dict with cached data if valid cache exists, else None.
        """
        if not os.path.exists(self.output_path):
            return None
        try:
            with open(self.output_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if not data.get('completed', False):
            return None

        if required_N_grid is not None:
            stored_N = data.get('parameters', {}).get('N_grid', None)
            if stored_N != required_N_grid:
                return None

        return data

    def check_resume(self):
        """Return checkpoint data if .checkpoint.json exists, else None."""
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        if data.get('completed', False):
            return None  # Already finished, not a resumable checkpoint

        return data

    def save_checkpoint(self, state_dict):
        """Atomic write of checkpoint. state_dict should include field arrays."""
        state_dict['completed'] = False
        atomic_write_json(state_dict, self.checkpoint_path)

    def should_checkpoint(self, current_time):
        """Check if enough sim-time has elapsed to warrant a checkpoint."""
        if current_time - self._last_checkpoint_time >= self.checkpoint_interval:
            self._last_checkpoint_time = current_time
            return True
        return False

    def finalize(self, results):
        """Write final output (no field arrays), delete checkpoint."""
        # Strip field arrays from final output
        clean = dict(results)
        for key in ('phi_b64', 'phi_dot_b64', 'field_state'):
            clean.pop(key, None)
        clean['completed'] = True

        atomic_write_json(clean, self.output_path)

        # Remove checkpoint file
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    def delete_checkpoint(self):
        """Remove checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)


def run_with_checkpointing(evolver, config, output_path, checkpoint_interval=50.0,
                           extra_diagnostic_fn=None):
    """Run simulation with automatic checkpointing, caching, and resume.

    Args:
        evolver: initialized SexticEvolver with IC already set
        config: dict with at least 'params' (containing N_grid, dt, T_final),
                'name' (string label), and any other metadata
        output_path: path for final JSON
        checkpoint_interval: time units between checkpoints (default 50.0)
        extra_diagnostic_fn: callable(evolver) -> dict, called at each
            recording point; results stored in output['extra_diagnostics']

    Returns:
        Final results dict (same as what gets written to output_path).
    """
    mgr = CheckpointManager(output_path, checkpoint_interval)
    N_grid = config.get('params', {}).get('N_grid', config.get('params', {}).get('N', 64))
    tag = config.get('name', '')

    # --- Cache check ---
    cached = mgr.check_cache(required_N_grid=N_grid)
    if cached is not None:
        print("CACHED: %s (N=%d)" % (tag, N_grid))
        sys.stdout.flush()
        return cached

    # --- Resume check ---
    resume_data = mgr.check_resume()
    if resume_data is not None:
        print("RESUMING: %s from T=%.1f" % (tag, resume_data.get('t', 0)))
        sys.stdout.flush()

    # --- Run evolution ---
    dt = config['params'].get('dt', 0.05)
    T_final = config['params'].get('T_final', 500.0)
    n_steps = int(T_final / dt)
    record_every = config.get('record_every', 10)
    print_every = config.get('print_every', 2000)

    def checkpoint_callback(state):
        mgr.save_checkpoint(state)

    # Determine checkpoint_every in steps from time interval
    steps_per_interval = max(1, int(checkpoint_interval / dt))

    state = evolver.evolve(
        dt=dt,
        n_steps=n_steps,
        record_every=record_every,
        checkpoint_every=steps_per_interval,
        checkpoint_callback=checkpoint_callback,
        resume_from=resume_data,
        print_every=print_every,
        tag=tag,
        extra_diagnostic_fn=extra_diagnostic_fn,
    )

    # --- Build final result ---
    results = {
        'completed': True,
        'metadata': config.get('metadata', {}),
        'parameters': config.get('params', {}),
        'initial_conditions': config.get('initial_conditions', {}),
        'time_series': state['time_series'],
        'final_state': {
            'E_total_0': state['E0'],
            'E_total_final': state['time_series']['E_total'][-1],
            'max_phi0': state['max_phi0'],
            'max_amplitude_final': state['time_series']['max_amplitude'][-1],
            'amplitude_retention': (
                state['time_series']['max_amplitude'][-1] / state['max_phi0']
                if state['max_phi0'] > 0 else 0.0
            ),
            'energy_drift_pct': (
                abs(state['time_series']['E_total'][-1] - state['E0'])
                / (abs(state['E0']) + 1e-30)
            ),
            'wall_seconds': state['wall_elapsed'],
        },
    }

    # Include extra diagnostics if present
    if 'extra_diagnostics' in state:
        results['extra_diagnostics'] = state['extra_diagnostics']

    # Merge any extra fields from config
    if 'name' in config:
        results['metadata']['config_name'] = config['name']

    mgr.finalize(results)
    print("COMPLETED: %s (%.1f min)" % (tag, state['wall_elapsed'] / 60.0))
    sys.stdout.flush()
    return results


def cleanup_study(output_dir):
    """Remove checkpoint and temp files after study completes."""
    output_dir = str(output_dir)
    checkpoints = glob.glob(os.path.join(output_dir, '*.checkpoint*'))
    temps = glob.glob(os.path.join(output_dir, '*.tmp'))

    total_freed = 0
    for f in checkpoints + temps:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            total_freed += size
        except OSError:
            pass

    n_ckpt = len(checkpoints)
    n_tmp = len(temps)
    if n_ckpt or n_tmp:
        print("Cleanup: removed %d checkpoints, %d temp files, freed %.1f MB"
              % (n_ckpt, n_tmp, total_freed / 1e6))

    # Verify all final outputs have completed: true
    jsons = glob.glob(os.path.join(output_dir, '*.json'))
    incomplete = []
    for fp in jsons:
        if '.checkpoint' in fp:
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
            if not data.get('completed', False):
                incomplete.append(os.path.basename(fp))
        except (json.JSONDecodeError, OSError):
            incomplete.append(os.path.basename(fp) + ' (unreadable)')

    if incomplete:
        print("WARNING: %d incomplete outputs: %s" % (len(incomplete), ', '.join(incomplete)))
    else:
        n_complete = len([j for j in jsons if '.checkpoint' not in j])
        print("All %d output files verified complete." % n_complete)
