#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated experiment runner for MultiDrone + BiRRT-Connect with multiple visualization modes.

Highlights
- Skips 'empty' by default: patterns = ['sparse','medium','narrow','cluttered']
- --viz-auto-live shows a real window NON-BLOCKING and auto-closes after --viz-duration seconds
- --viz-once-per-pattern shows at most one window per pattern
- --viz-auto saves offscreen screenshots (no windows)
- --viz is the original interactive window (press ENTER to close)
- Optional forced blocking wall across the corridor so obstacles are in the way
- Optional environment validation and "ensure all shapes" (box/sphere/cylinder) for non-empty maps
"""

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from rrt_connect import birrt_connect, build_goal_from_centers
import multi_drone as md
from multi_drone import MultiDrone

# ---------------------------- Plotting helpers ----------------------------
def _no_plot(self):
    self._plotter = None
    self._drone_visuals = []
    return

def _init_plot_offscreen(self):
    from vedo import Plotter, Line, Sphere, Cylinder
    self._plotter = Plotter(interactive=False, offscreen=True)
    self._drone_visuals = []
    for i in range(self.N):
        body = Sphere(r=0.1).c('cyan')
        arm1 = Cylinder(r=0.03, height=1.0).c('black')
        arm2 = Cylinder(r=0.03, height=1.0).c('black')
        pts = np.array(self.trajectories[i]) if hasattr(self, 'trajectories') else np.array([self.initial_configuration[i]])
        traj = Line(pts).lw(2).c('blue')
        self._drone_visuals.append((body, arm1, arm2, traj))
    visuals = []
    for i in range(self.N):
        visuals.extend(self._drone_visuals[i])
    visuals.extend(self._obstacles_viz)
    visuals.extend(self._goal_viz)
    self._plotter.show(*visuals, axes=dict(xrange=(0, 50), yrange=(0, 50), zrange=(0, 50),
                                           xygrid=True, yzgrid=True, zxgrid=True),
                       viewup='z', interactive=False, mode=8)

def _init_plot_onscreen_nonblocking(self):
    from vedo import Plotter, Line, Sphere, Cylinder
    self._plotter = Plotter(interactive=False, offscreen=False)
    self._drone_visuals = []
    for i in range(self.N):
        body = Sphere(r=0.1).c('cyan')
        arm1 = Cylinder(r=0.03, height=1.0).c('black')
        arm2 = Cylinder(r=0.03, height=1.0).c('black')
        pts = np.array(self.trajectories[i]) if hasattr(self, 'trajectories') else np.array([self.initial_configuration[i]])
        traj = Line(pts).lw(2).c('blue')
        self._drone_visuals.append((body, arm1, arm2, traj))
    visuals = []
    for i in range(self.N):
        visuals.extend(self._drone_visuals[i])
    visuals.extend(self._obstacles_viz)
    visuals.extend(self._goal_viz)
    self._plotter.show(*visuals, axes=dict(xrange=(0, 50), yrange=(0, 50), zrange=(0, 50),
                                           xygrid=True, yzgrid=True, zxgrid=True),
                       viewup='z', interactive=False, mode=8)

# ---------------------------- CI utilities ----------------------------
def mean_ci_95(x: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(list(x), dtype=float)
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    m = float(np.mean(arr)); s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return (m, 1.96 * s / math.sqrt(max(1, len(arr))))

def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return (float('nan'), float('nan'))
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    half = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4*n**2)))
    return center, half

# ---------------------------- Env generation ----------------------------
def make_initial_goals(bounds: np.ndarray, K: int, r_drone: float = 0.3) -> Tuple[List[List[float]], List[Dict]]:
    x0, x1 = bounds[0]; y0, y1 = bounds[1]; z0, z1 = bounds[2]
    margin = 2.0; spacing = max(2.5 * r_drone, 1.2)
    inits, goals = [], []
    for i in range(K):
        inits.append([float(x0 + margin + i*spacing), float(y0 + margin), float(z0 + 1.0)])
        goals.append({'position': [float(x1 - margin - i*spacing), float(y1 - margin), float(z0 + 1.0)], 'radius': 1.0})
    return inits, goals

def add_box(center, size, color='red', rotation=None) -> Dict:
    return {'type': 'box', 'position': list(map(float, center)), 'size': list(map(float, size)),
            'rotation': rotation or [0,0,0], 'color': color}
def add_sphere(center, radius, color='red') -> Dict:
    return {'type': 'sphere', 'position': list(map(float, center)), 'radius': float(radius), 'color': color}
def add_cylinder(p1, p2, radius, color='red', rotation=None) -> Dict:
    return {'type': 'cylinder', 'endpoints': [list(map(float, p1)), list(map(float, p2))],
            'radius': float(radius), 'rotation': rotation or [0,0,0], 'color': color}

def add_blocking_wall(env: Dict, gap_width: float, wall_thickness: float, gap_offset: float) -> None:
    bx = env['bounds']['x']; by = env['bounds']['y']; bz = env['bounds']['z']
    x0, x1 = float(bx[0]), float(bx[1]); y0, y1 = float(by[0]), float(by[1]); z0, z1 = float(bz[0]), float(bz[1])
    cx = 0.5 * (x0 + x1)
    inits = np.array(env['initial_configuration'], dtype=np.float32)
    goals_centers = np.array([g['position'] for g in env['goals']], dtype=np.float32)
    p_start = np.mean(inits, axis=0); p_goal = np.mean(goals_centers, axis=0)
    dx = p_goal[0] - p_start[0]
    t = 0.5 if abs(dx) < 1e-6 else max(0.0, min(1.0, (cx - p_start[0]) / dx))
    y_line = float(p_start[1] + t * (p_goal[1] - p_start[1]))
    y_gap = max(y0 + gap_width/2 + 0.1, min(y1 - gap_width/2 - 0.1, y_line + gap_offset))
    sz = (z1 - z0)
    lower_sy = max(0.0, (y_gap - gap_width/2.0) - y0)
    if lower_sy > 0.2:
        env['obstacles'].append(add_box((cx, y0 + lower_sy/2.0, z0 + sz/2.0), (wall_thickness, lower_sy, sz)))
    upper_sy = max(0.0, y1 - (y_gap + gap_width/2.0))
    if upper_sy > 0.2:
        env['obstacles'].append(add_box((cx, y1 - upper_sy/2.0, z0 + sz/2.0), (wall_thickness, upper_sy, sz)))

def gen_env_dict(pattern: str, K: int, seed: int = 0, bounds=((0,50),(0,50),(0,50))) -> Dict:
    rng = random.Random(seed)
    bx, by, bz = bounds
    inits, goals = make_initial_goals(np.array(bounds, dtype=np.float32), K)
    obstacles: List[Dict] = []
    cx = 0.5 * (bx[0] + bx[1]); cy = 0.5 * (by[0] + by[1])
    if pattern == 'sparse':
        obstacles = [add_box((cx - 5, cy, 1.5), (6,6,3)),
                     add_sphere((cx + 6, cy - 8, 2.0), 2.5),
                     add_cylinder((cx - 12, cy + 8, 0.0), (cx - 12, cy + 8, 10.0), 1.8)]
    elif pattern == 'medium':
        obstacles = [add_box((cx - 7, cy, 1.5), (8,6,3)),
                     add_box((cx + 10, cy + 5, 1.5), (6,6,3)),
                     add_sphere((cx + 2, cy - 10, 2.0), 3.0),
                     add_cylinder((cx - 12, cy + 8, 0.0), (cx - 12, cy + 8, 12.0), 2.0)]
    elif pattern == 'narrow':
        obstacles = [add_box((cx - 3, cy, 2.0), (20,18,4)),
                     add_box((cx + 3, cy, 2.0), (20,18,4)),
                     add_sphere((cx + 6, cy + 10, 1.5), 2.2),
                     add_cylinder((cx, cy + 15, 0.0), (cx, cy + 15, 12.0), 1.8)]
    elif pattern == 'cluttered':
        for _ in range(6):
            shape = rng.choice(['box','sphere','cylinder'])
            px = rng.uniform(bx[0]+8, bx[1]-8); py = rng.uniform(by[0]+8, by[1]-8)
            if shape == 'box':
                sx = rng.uniform(3,8); sy = rng.uniform(3,8); sz = rng.uniform(2,5)
                obstacles.append(add_box((px, py, sz/2.0), (sx, sy, sz)))
            elif shape == 'sphere':
                r = rng.uniform(1.5,3.5)
                obstacles.append(add_sphere((px, py, r+0.5), r))
            else:
                h = rng.uniform(6,14); r = rng.uniform(1.2,2.5)
                obstacles.append(add_cylinder((px, py, 0.0), (px, py, h), r))
        types = {o['type'] for o in obstacles}
        if 'box' not in types: obstacles.append(add_box((cx-6, cy-4, 2.0), (6,6,4)))
        if 'sphere' not in types: obstacles.append(add_sphere((cx+2, cy+6, 2.5), 2.5))
        if 'cylinder' not in types: obstacles.append(add_cylinder((cx-10, cy+10, 0.0), (cx-10, cy+10, 12.0), 2.0))
    else:
        raise ValueError(f"Unknown pattern {pattern}")
    return {'bounds': {'x': list(bounds[0]), 'y': list(bounds[1]), 'z': list(bounds[2])},
            'initial_configuration': inits, 'obstacles': obstacles, 'goals': goals}

def write_env_yaml(env: Dict, path: str) -> None:
    with open(path, 'w') as f:
        yaml.safe_dump(env, f, sort_keys=False)

# ---------------------------- Metrics ----------------------------
@dataclass
class TrialResult:
    exp_type: str; pattern: str; K: int; trial: int; seed: int
    success: int; time_sec: float; waypoints: int; path_len_total: float; path_len_max_per_drone: float

def path_lengths(path: List[np.ndarray]) -> Tuple[float, float]:
    if path is None or len(path) < 2: return (float('nan'), float('nan'))
    K = path[0].shape[0]; per = np.zeros(K, dtype=float)
    for s in range(1, len(path)): per += np.linalg.norm(path[s]-path[s-1], axis=1)
    return float(np.sum(per)), float(np.max(per))

# ---------------------------- Runner ----------------------------
def run_trials_for_env(env_yaml: str, exp_type: str, pattern: str, K: int, trials: int,
                       time_limit: float, delta_pos: float, goal_bias: float, smoothing_trials: int,
                       seed_base: int, verbose: bool, summary_mode: str, summary_interval: int,
                       viz: bool, viz_auto: bool, viz_auto_live: bool, viz_duration: float,
                       viz_once_per_pattern: bool, viz_dir: Optional[str]) -> List[TrialResult]:
    sim = MultiDrone(num_drones=K, environment_file=env_yaml)
    q_start = sim.initial_configuration.astype(np.float32)
    q_goal = build_goal_from_centers(sim).astype(np.float32)

    results: List[TrialResult] = []
    shown_this_pattern = False

    for t in range(trials):
        seed = seed_base + t
        if verbose: print(f"[{exp_type}] pattern={pattern} K={K} trial {t+1}/{trials} seed={seed} ... ", end='', flush=True)
        t0 = time.time()
        path = birrt_connect(sim, q_start=q_start, q_goal=q_goal, time_limit=time_limit,
                             delta_pos=delta_pos, goal_bias=goal_bias, smoothing_trials=smoothing_trials, rng_seed=seed)
        dt = time.time() - t0

        if path is None:
            success = 0; wpts = 0; Ltot = Lmax = float('nan')
            if verbose: print(f"FAIL time={dt:.2f}s", flush=True)
        else:
            success = 1 if sim.is_goal(path[-1]) else 0
            wpts = len(path); Ltot, Lmax = path_lengths(path)
            if verbose:
                status = 'OK' if success == 1 else 'NOT_GOAL'
                print(f"{status} time={dt:.2f}s wpts={wpts} Ltot={Ltot:.2f} Lmax={Lmax:.2f}", end='', flush=True)

            if success == 1 and (viz or viz_auto or viz_auto_live) and (not viz_once_per_pattern or not shown_this_pattern):
                try:
                    if viz_auto:
                        if viz_dir is not None:
                            os.makedirs(viz_dir, exist_ok=True)
                            fname = os.path.join(viz_dir, f"{exp_type}_{pattern}_K{K}_trial{t+1}.png")
                        else:
                            fname = f"{exp_type}_{pattern}_K{K}_trial{t+1}.png"
                        if hasattr(sim, "_plotter") and sim._plotter is not None and hasattr(sim._plotter, "interactive"):
                            _old = sim._plotter.interactive
                            sim._plotter.interactive = lambda *a, **k: None
                            try: sim.visualize_paths(path)
                            finally:
                                try:
                                    sim._plotter.screenshot(fname)
                                    if verbose: print(f" [saved {fname}]", end='')
                                except Exception as ee:
                                    if verbose: print(f" [screenshot failed: {ee}]", end='')
                                sim._plotter.interactive = _old
                        else:
                            sim.visualize_paths(path)
                    elif viz_auto_live:
                        import time as _time
                        if hasattr(sim, "_plotter") and sim._plotter is not None:
                            _old = getattr(sim._plotter, 'interactive', None)
                            sim._plotter.interactive = lambda *a, **k: None
                        sim.visualize_paths(path)
                        _time.sleep(max(0.0, float(viz_duration)))
                        if hasattr(sim, "_plotter") and sim._plotter is not None:
                            try: sim._plotter.close()
                            except Exception: pass
                            if _old is not None: sim._plotter.interactive = _old
                    else:
                        sim.visualize_paths(path)
                    shown_this_pattern = True
                except Exception as e:
                    if verbose: print(f" [viz error: {e}]", end='')
            if verbose: print('', flush=True)

        results.append(TrialResult(exp_type, pattern, K, t, seed, success, dt, wpts, Ltot, Lmax))

        if verbose:
            do_print = (summary_mode == 'pattern' and (t+1)==trials) or                        (summary_mode == 'every' and (((t+1)%max(1,summary_interval))==0 or (t+1)==trials))
            if do_print:
                n = len(results); succ = sum(r.success for r in results)
                p = succ/n if n else 0.0; t_mean = sum(r.time_sec for r in results)/n if n else float('nan')
                print(f"[summary] {exp_type} pattern={pattern} K={K} {succ}/{n} success ({p:.1%}), mean time={t_mean:.2f}s", flush=True)

    return results

# ---------------------------- Plots and summary ----------------------------
def save_results_csv(rows: List[TrialResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['exp_type','pattern','K','trial','seed','success','time_sec','waypoints','path_len_total','path_len_max_per_drone'])
        for r in rows:
            w.writerow([r.exp_type, r.pattern, r.K, r.trial, r.seed, r.success, f"{r.time_sec:.6f}", r.waypoints,
                        f"{r.path_len_total:.6f}" if not math.isnan(r.path_len_total) else '',
                        f"{r.path_len_max_per_drone:.6f}" if not math.isnan(r.path_len_max_per_drone) else ''])

def summarize_and_save(rows: List[TrialResult], group_key: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    groups: Dict[str, List[TrialResult]] = {}
    for r in rows: groups.setdefault(str(getattr(r, group_key)), []).append(r)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow([group_key,'n_trials','success_rate_mean','success_rate_ci','time_mean','time_ci','wpts_mean','wpts_ci','Ltotal_mean','Ltotal_ci','Lmax_mean','Lmax_ci'])
        for k, lst in sorted(groups.items(), key=lambda kv: kv[0]):
            n = len(lst); p_hat = sum(r.success for r in lst)/max(1,n); p_c, p_hw = wilson_ci(p_hat, n)
            t_mean, t_ci = mean_ci_95([r.time_sec for r in lst])
            wpts = [r.waypoints for r in lst if r.success==1]; w_mean, w_ci = mean_ci_95(wpts) if wpts else (float('nan'), float('nan'))
            Lt = [r.path_len_total for r in lst if r.success==1 and not math.isnan(r.path_len_total)]
            Lm = [r.path_len_max_per_drone for r in lst if r.success==1 and not math.isnan(r.path_len_max_per_drone)]
            Lt_m, Lt_ci = mean_ci_95(Lt) if Lt else (float('nan'), float('nan'))
            Lm_m, Lm_ci = mean_ci_95(Lm) if Lm else (float('nan'), float('nan'))
            w.writerow([k, n, f"{p_c:.3f}", f"{p_hw:.3f}", f"{t_mean:.3f}", f"{t_ci:.3f}",
                        f"{w_mean:.2f}" if not math.isnan(w_mean) else '', f"{w_ci:.2f}" if not math.isnan(w_ci) else '',
                        f"{Lt_m:.2f}" if not math.isnan(Lt_m) else '', f"{Lt_ci:.2f}" if not math.isnan(Lt_ci) else '',
                        f"{Lm_m:.2f}" if not math.isnan(Lm_m) else '', f"{Lm_ci:.2f}" if not math.isnan(Lm_ci) else ''])

def try_make_plots(summary_csv: str, x_field: str, out_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as e:
        print(f"Plotting skipped (missing libraries): {e}"); return
    df = pd.read_csv(summary_csv); x = df[x_field].values
    y = df['success_rate_mean'].astype(float).values; ci = df['success_rate_ci'].astype(float).values
    plt.figure(); plt.errorbar(x, y, yerr=ci, fmt='-o'); plt.xlabel(x_field); plt.ylabel('Success rate'); plt.title('Success rate (95% CI)'); plt.grid(True); plt.tight_layout(); plt.savefig(out_prefix+'_success.png'); plt.close()
    y = df['time_mean'].astype(float).values; ci = df['time_ci'].astype(float).values
    plt.figure(); plt.errorbar(x, y, yerr=ci, fmt='-o'); plt.xlabel(x_field); plt.ylabel('Planning time (s)'); plt.title('Planning time (95% CI)'); plt.grid(True); plt.tight_layout(); plt.savefig(out_prefix+'_time.png'); plt.close()
    if 'Ltotal_mean' in df.columns:
        try:
            y = df['Ltotal_mean'].astype(float).values; ci = df['Ltotal_ci'].astype(float).values
            plt.figure(); plt.errorbar(x, y, yerr=ci, fmt='-o'); plt.xlabel(x_field); plt.ylabel('Total path length'); plt.title('Total path length (95% CI)'); plt.grid(True); plt.tight_layout(); plt.savefig(out_prefix+'_Ltotal.png'); plt.close()
        except Exception: pass

def main():
    parser = argparse.ArgumentParser(description='Automated experiments for BiRRT-Connect + MultiDrone')
    parser.add_argument('--outdir', type=str, default='results', help='Output directory')
    parser.add_argument('--trials', type=int, default=30, help='Trials per setting')
    parser.add_argument('--time-limit', type=float, default=30.0, help='Time limit per trial (s)')
    parser.add_argument('--delta-pos', type=float, default=0.75, help='Steering step (m)')
    parser.add_argument('--goal-bias', type=float, default=0.1, help='Goal bias probability')
    parser.add_argument('--smooth-trials', type=int, default=200, help='Shortcut smoothing iterations')
    parser.add_argument('--seed-base', type=int, default=1234, help='Base RNG seed')

    parser.add_argument('--env-suite', action='store_true', help='Run environment complexity sweep')
    parser.add_argument('--k-suite', action='store_true', help='Run team-size sweep')

    parser.add_argument('--k-fixed', type=int, default=3, help='Number of drones for env sweep')
    parser.add_argument('--patterns', nargs='*', default=['sparse','medium','narrow','cluttered'], help='Patterns to evaluate (default skips empty)')

    parser.add_argument('--k-values', nargs='*', type=int, default=[1,3,6,9,12], help='Team sizes for K sweep')
    parser.add_argument('--k-env-pattern', type=str, default='medium', help='Pattern for team-size sweep')

    parser.add_argument('--make-plots', action='store_true', help='Generate PNG plots from summaries')
    parser.add_argument('--verbose', action='store_true', help='Print per-trial progress')
    parser.add_argument('--summary-mode', choices=['pattern','every'], default='pattern', help="'pattern' prints one summary per pattern; 'every' prints every N trials")
    parser.add_argument('--summary-interval', type=int, default=5, help='When --summary-mode=every, print every N trials')

    parser.add_argument('--viz', action='store_true', help='Show blocking windows (press ENTER to close)')
    parser.add_argument('--viz-auto', action='store_true', help='Offscreen screenshots (no windows)')
    parser.add_argument('--viz-auto-live', action='store_true', help='Onscreen non-blocking windows auto-close after duration')
    parser.add_argument('--viz-duration', type=float, default=3.0, help='Seconds to show each auto-live window')
    parser.add_argument('--viz-once-per-pattern', action='store_true', help='Show at most one window per pattern')
    parser.add_argument('--viz-dir', type=str, default='viz', help='Screenshot subdir for --viz-auto')

    parser.add_argument('--force-block', action='store_true', help='Insert blocking wall across the corridor')
    parser.add_argument('--gap-width', type=float, default=4.0, help='Gap width (m) in the blocking wall')
    parser.add_argument('--wall-thickness', type=float, default=2.0, help='Wall thickness along x (m)')
    parser.add_argument('--gap-offset', type=float, default=8.0, help='Offset of gap center from straight-line intersection (m)')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Plotting mode selection
    if args.viz_auto_live and not (args.viz or args.viz_auto):
        try: md.MultiDrone._init_plot = _init_plot_onscreen_nonblocking
        except Exception: pass
    elif args.viz_auto and not args.viz:
        try: md.MultiDrone._init_plot = _init_plot_offscreen
        except Exception: pass
    elif not (args.viz or args.viz_auto or args.viz_auto_live):
        try: md.MultiDrone._init_plot = _no_plot
        except Exception: pass

    all_rows: List[TrialResult] = []

    if args.env_suite:
        for pattern in args.patterns:
            env = gen_env_dict(pattern=pattern, K=args.k_fixed, seed=42)
            if args.force_block:
                add_blocking_wall(env, gap_width=args.gap_width, wall_thickness=args.wall_thickness, gap_offset=args.gap_offset)
            env_path = os.path.join(args.outdir, f"env_{pattern}_K{args.k_fixed}.yaml")
            write_env_yaml(env, env_path)

            rows = run_trials_for_env(env_path, 'complexity', pattern, args.k_fixed, args.trials,
                                      args.time_limit, args.delta_pos, args.goal_bias, args.smooth_trials,
                                      args.seed_base, args.verbose, args.summary_mode, args.summary_interval,
                                      args.viz, args.viz_auto, args.viz_auto_live, args.viz_duration,
                                      args.viz_once_per_pattern, os.path.join(args.outdir, args.viz_dir))
            all_rows.extend(rows)

        raw_csv = os.path.join(args.outdir, 'raw_env_suite.csv'); save_results_csv(all_rows, raw_csv)
        summary_csv = os.path.join(args.outdir, 'summary_env_suite.csv'); summarize_and_save([r for r in all_rows if r.exp_type=='complexity'], 'pattern', summary_csv)
        if args.make_plots: try_make_plots(summary_csv, x_field='pattern', out_prefix=os.path.join(args.outdir, 'env_suite'))

    if args.k_suite:
        rows_k: List[TrialResult] = []
        for K in args.k_values:
            env = gen_env_dict(pattern=args.k_env_pattern, K=K, seed=99)
            if args.force_block:
                add_blocking_wall(env, gap_width=args.gap_width, wall_thickness=args.wall_thickness, gap_offset=args.gap_offset)
            env_path = os.path.join(args.outdir, f"env_{args.k_env_pattern}_K{K}.yaml"); write_env_yaml(env, env_path)

            rows = run_trials_for_env(env_path, 'team_size', args.k_env_pattern, K, args.trials,
                                      args.time_limit, args.delta_pos, args.goal_bias, args.smooth_trials,
                                      args.seed_base, args.verbose, args.summary_mode, args.summary_interval,
                                      args.viz, args.viz_auto, args.viz_auto_live, args.viz_duration,
                                      args.viz_once_per_pattern, os.path.join(args.outdir, args.viz_dir))
            rows_k.extend(rows)

        raw_csv_k = os.path.join(args.outdir, 'raw_k_suite.csv'); save_results_csv(rows_k, raw_csv_k)
        summary_csv_k = os.path.join(args.outdir, 'summary_k_suite.csv'); summarize_and_save(rows_k, 'K', summary_csv_k)
        if args.make_plots: try_make_plots(summary_csv_k, x_field='K', out_prefix=os.path.join(args.outdir, 'k_suite'))

if __name__ == '__main__':
    main()
