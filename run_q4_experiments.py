#!/usr/bin/env python3
"""
Part B, Question 4: Empirical study vs. environment complexity (+ optional K sweep).

This script generates environments of increasing complexity, runs the
RRT-Connect planner from rrt_connect_multidrone.py using the provided
MultiDrone simulator, and reports summary statistics (mean and 95% CI).

NEW: You can sweep the number of drones K, e.g. 1..12, via --k-list or --k-range.
Default behaviour keeps K fixed (inferred from environment.yaml), focusing on Q4's
environment-complexity study.

Usage examples:
    # Q4 default: vary complexity levels, keep K from environment.yaml
    python run_q4_experiments.py --base-env environment.yaml --levels 0 1 2 3 4 --seeds 0 1 2 3 4 --outdir results_q4

    # Sweep K from 1 to 12 as well (optional)
    python run_q4_experiments.py --base-env environment.yaml --levels 0 1 2 3 4 --seeds 0 1 2 3 4 \
        --k-range 1 12 --outdir results_q4_k

Outputs:
- CSV with per-trial metrics: success, time, nodes, iterations, path_len, level, seed, K
- JSON with summary stats by (K, level): means and 95% CIs
"""
import argparse, os, time, math, json, random
import numpy as np
import yaml
from typing import Dict, Any, Tuple, List

from multi_drone import MultiDrone
from rrt_connect_multidrone import rrt_connect_plan

# --------------------------- Helpers ---------------------------

def read_base(env_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bounds 3x2, initial Mx3, goals Mx3) from a base YAML file; M may differ from K later."""
    with open(env_path, 'r') as f:
        data = yaml.safe_load(f)
    bounds = np.array([data['bounds']['x'], data['bounds']['y'], data['bounds']['z']], dtype=np.float32)
    init = np.array(data['initial_configuration'], dtype=np.float32)
    goals = np.array([g['position'] for g in data['goals']], dtype=np.float32)
    return bounds, init, goals

def randf(lo, hi, rng) -> float:
    return float(rng.uniform(lo, hi))

def make_box(position, size, rotation=(0,0,0), color='red'):
    return {'type':'box', 'position': [float(x) for x in position], 'size': [float(x) for x in size], 'rotation': list(rotation), 'color': color}

def make_sphere(position, radius, rotation=(0,0,0), color='red'):
    return {'type':'sphere', 'position': [float(x) for x in position], 'radius': float(radius), 'rotation': list(rotation), 'color': color}

def make_cylinder(p1, p2, radius, rotation=(0,0,0), color='red'):
    return {'type':'cylinder', 'endpoints': [[float(x) for x in p1], [float(x) for x in p2]], 'radius': float(radius), 'rotation': list(rotation), 'color': color}

def make_init_goals_for_K(bounds: np.ndarray, base_init: np.ndarray, base_goals: np.ndarray, K: int, rng: random.Random, min_sep: float=0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Kx3 start/goal sets with >= min_sep pairwise spacing.
    If K <= M (base count), take first K then (if needed) spread slightly to reach min_sep.
    If K > M, sample additional positions via simple Poisson-disk attempts inside bounds,
    near the centroid of the base sets.
    """
    import numpy as _np
    M = base_init.shape[0]
    # helper spacing check
    def ok(arr):
        if len(arr) < 2: return True
        D = _np.linalg.norm(arr[:,None,:]-arr[None,:,:], axis=-1)
        mask = _np.triu(_np.ones((len(arr),len(arr))),1).astype(bool)
        return not _np.any(D[mask] < min_sep)

    # clamp helper
    def clamp(arr):
        lo = bounds[:,0] + 0.5
        hi = bounds[:,1] - 0.5
        return _np.minimum(hi, _np.maximum(lo, arr))

    # Start with as many as the base contains
    init = base_init[:min(K,M)].copy()
    goals = base_goals[:min(K,M)].copy()

    # If we need more, sample around centroids
    smean = _np.mean(base_init, axis=0)
    gmean = _np.mean(base_goals, axis=0)

    def sample_around(center):
        # random ring in XY, z from the center's z
        r = rng.uniform(min_sep*0.6, min_sep*1.2)
        ang = rng.uniform(0, 2*_np.pi)
        return _np.array([center[0] + r*_np.cos(ang),
                          center[1] + r*_np.sin(ang),
                          center[2]], dtype=_np.float32)

    while len(init) < K:
        cand = sample_around(smean)
        cand_g = sample_around(gmean)
        iters = 0
        while iters < 200:
            iters += 1
            cand = clamp(cand[None,:])[0]
            cand_g = clamp(cand_g[None,:])[0]
            if ok(_np.vstack([init, cand])) and ok(_np.vstack([goals, cand_g])):
                break
            # resample
            cand = sample_around(smean)
            cand_g = sample_around(gmean)
        if iters >= 200:
            # fall back: place on a circle growing outward
            radius = (len(init)+1) * (min_sep*0.6)
            ang = rng.uniform(0, 2*_np.pi)
            cand = clamp(_np.array([smean[0] + radius*_np.cos(ang), smean[1] + radius*_np.sin(ang), smean[2]], dtype=_np.float32)[None,:])[0]
            cand_g = clamp(_np.array([gmean[0] + radius*_np.cos(ang), gmean[1] + radius*_np.sin(ang), gmean[2]], dtype=_np.float32)[None,:])[0]
        init = _np.vstack([init, cand])
        goals = _np.vstack([goals, cand_g])

    # If K <= M but spacing too tight, push apart slightly
    for arr in (init, goals):
        if not ok(arr):
            center = arr.mean(axis=0)
            for _ in range(50):
                if ok(arr): break
                for i in range(len(arr)):
                    v = arr[i]-center
                    if (v[:2]**2).sum() < 1e-8:
                        v[:2] = _np.array([rng.uniform(-1,1), rng.uniform(-1,1)], dtype=_np.float32)
                    arr[i,:2] = arr[i,:2] + (v[:2] / (1e-9+_np.linalg.norm(v[:2])))*0.05
                arr[:] = clamp(arr)

    return init.astype(_np.float32), goals.astype(_np.float32)
def generate_env(bounds: np.ndarray, init: np.ndarray, goals: np.ndarray, level: int, seed: int) -> Dict[str, Any]:
    """
    Produce a YAML dict representing an environment of a given complexity level.

    Complexity dial:
      L0  : free space (0 obstacles)
      L1  : sparse clutter (2-3 obstacles) wide passages
      L2  : moderate clutter (4-5 obstacles)
      L3  : heavy clutter (6-8 obstacles)
      L4  : narrow passage (two "walls" with a small gap) + sparse clutter
    """
    rng = random.Random(seed)
    env = {
        'bounds': {'x': bounds[0].tolist(), 'y': bounds[1].tolist(), 'z': bounds[2].tolist()},
        'initial_configuration': init.tolist(),
        'obstacles': [],
        'goals': [{'position': g.tolist(), 'radius': 1.0} for g in goals],
    }

    # Helper: keep clutter away from mean start/goal areas
    def far_from_start_goal(pos):
        smean = np.mean(init, axis=0)
        gmean = np.mean(goals, axis=0)
        return (np.linalg.norm(pos - smean) > 3.0) and (np.linalg.norm(pos - gmean) > 3.0)

    XR = (bounds[0,0]+2.0, bounds[0,1]-2.0)
    YR = (bounds[1,0]+2.0, bounds[1,1]-2.0)
    ZR = (bounds[2,0]+0.5, min(bounds[2,1]-0.5, bounds[2,0]+10.0))

    def add_random_obstacle():
        typ = rng.choice(['box','sphere','cylinder'])
        if typ == 'box':
            pos = np.array([randf(*XR, rng), randf(*YR, rng), randf(*ZR, rng)], dtype=np.float32)
            if not far_from_start_goal(pos): return
            size = np.array([randf(1.5,4.5,rng), randf(1.5,4.5,rng), randf(1.0,3.0,rng)], dtype=np.float32)
            env['obstacles'].append(make_box(pos, size))
        elif typ == 'sphere':
            pos = np.array([randf(*XR, rng), randf(*YR, rng), randf(*ZR, rng)], dtype=np.float32)
            if not far_from_start_goal(pos): return
            radius = randf(1.0, 2.5, rng)
            env['obstacles'].append(make_sphere(pos, radius))
        else:
            p1 = np.array([randf(*XR, rng), randf(*YR, rng), bounds[2,0]], dtype=np.float32)
            p2 = p1.copy(); p2[2] = min(bounds[2,1], p1[2] + randf(8.0, 15.0, rng))
            radius = randf(1.0, 2.5, rng)
            env['obstacles'].append(make_cylinder(p1, p2, radius))

    if level == 0:
        pass
    elif level == 1:
        for _ in range(rng.randint(2,3)):
            add_random_obstacle()
    elif level == 2:
        for _ in range(rng.randint(4,5)):
            add_random_obstacle()
    elif level == 3:
        for _ in range(rng.randint(6,8)):
            add_random_obstacle()
    elif level == 4:
        # Pair of walls with a small gap (narrow passage)
        x_mid = (bounds[0,0] + bounds[0,1]) * 0.5
        y_mid = (bounds[1,0] + bounds[1,1]) * 0.5
        gap_width = 1.0  # tighten to increase difficulty
        # Left wall
        left_len = (x_mid - gap_width*0.5) - (bounds[0,0]+1.0)
        if left_len > 0.5:
            left_center = np.array([bounds[0,0]+1.0 + left_len*0.5, y_mid, 1.0], dtype=np.float32)
            left_size   = np.array([left_len, 2.0, 2.0], dtype=np.float32)
            env['obstacles'].append(make_box(left_center, left_size, color='red'))
        # Right wall
        right_len = (bounds[0,1]-1.0) - (x_mid + gap_width*0.5)
        if right_len > 0.5:
            right_center = np.array([x_mid + gap_width*0.5 + right_len*0.5, y_mid, 1.0], dtype=np.float32)
            right_size   = np.array([right_len, 2.0, 2.0], dtype=np.float32)
            env['obstacles'].append(make_box(right_center, right_size, color='red'))
        # Add sparse clutter
        for _ in range(2):
            add_random_obstacle()
    else:
        raise ValueError("Unsupported complexity level")

    return env

def mean_ci95(samples: List[float]) -> Tuple[float, float]:
    """Return (mean, halfwidth) where CI95 = mean ± halfwidth (normal approx)."""
    if len(samples) == 0:
        return (float('nan'), float('nan'))
    m = float(np.mean(samples))
    s = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    hw = 1.96 * s / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return m, hw

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-env', type=str, default='environment.yaml')
    ap.add_argument('--levels', type=int, nargs='+', default=[0,1,2,3,4])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--k', type=int, default=None, help='Number of drones (single K). Default: infer from base env.')
    ap.add_argument('--k-list', type=int, nargs='+', default=None, help='Run a sweep over these K values (e.g., 1 2 3 ... 12).')
    ap.add_argument('--k-range', type=int, nargs=2, default=None, help='Inclusive range start end to sweep K.')
    ap.add_argument('--time-limit', type=float, default=120.0)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--goal-bias', type=float, default=0.25)
    ap.add_argument('--outdir', type=str, default='results_q4')
    ap.add_argument('--viz', type=int, default=0)
    ap.add_argument('--drone-radius', type=float, default=0.3,
                    help='Assumed drone radius (m). Used for spacing checks (default 0.3).')
    ap.add_argument('--min-sep', type=float, default=None,
                    help='Minimum allowed inter-drone center distance. Default: 2*drone_radius + 0.1')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read base
    with open(args.base_env, 'r') as f:
        base = yaml.safe_load(f)
    k_base = len(base['initial_configuration'])

    bounds, base_init, base_goals = read_base(args.base_env)

    # Min separation
    min_sep = (2.0*args.drone_radius + 0.1) if args.min_sep is None else float(args.min_sep)

    # Decide K sweep
    if args.k_list is not None:
        k_values = args.k_list
    elif args.k_range is not None:
        start, end = args.k_range
        k_values = list(range(start, end+1))
    else:
        k_values = [args.k or k_base]

    # Results
    per_trial = []

    for K in k_values:
        for level in args.levels:
            for seed in args.seeds:
                rng = random.Random(seed + 1000*K + 100*level)
                init, goals = make_init_goals_for_K(bounds, base_init, base_goals, K, rng, min_sep=min_sep)
                # Try up to 10 times to generate a map where the initial configuration is valid (and spaced)
                ok_init = False
                for attempt in range(10):
                    env = generate_env(bounds, init, goals, level, seed + 10000*attempt)
                    env_path = os.path.join(args.outdir, f'env_K{K}_level{level}_seed{seed}_try{attempt}.yaml')
                    with open(env_path, 'w') as f:
                        yaml.safe_dump(env, f, sort_keys=False)
                    sim = MultiDrone(num_drones=K, environment_file=env_path)
                    # quick spacing guard too
                    import numpy as _np
                    def _ok(arr):
                        if len(arr) < 2: return True
                        D = _np.linalg.norm(arr[:,None,:]-arr[None,:,:], axis=-1)
                        mask = _np.triu(_np.ones((len(arr),len(arr))),1).astype(bool)
                        return not _np.any(D[mask] < min_sep)
                    if sim.is_valid(init) and _ok(init):
                        ok_init = True
                        break
                if not ok_init:
                    print(f'[WARN] Could not produce a valid initial configuration after 10 attempts for K={K}, level={level}, seed={seed}. Skipping trial.')
                    continue

                # Run planner
                t0 = time.time()
                ok, path_flat, stats = rrt_connect_plan(
                    sim,
                    time_limit=args.time_limit,
                    step=args.step,
                    goal_bias=args.goal_bias,
                    seed=seed,
                    do_postprocess=True,
                )
                elapsed = time.time() - t0

                # Path length in joint space
                path_len = (sum(float(np.linalg.norm(a-b)) for a,b in zip(path_flat[:-1], path_flat[1:]))
                            if ok and len(path_flat) >= 2 else float('nan'))

                per_trial.append({
                    'K': K,
                    'level': level,
                    'seed': seed,
                    'success': int(ok),
                    'time_sec': stats.get('time', elapsed),
                    'nodes': stats.get('nodes', None),
                    'iterations': stats.get('iterations', None),
                    'path_len_joint': path_len,
                })

                # Optional viz for a representative success
                if args.viz and ok:
                    path_cfgs = [p.reshape(K,3).astype(np.float32) for p in path_flat]
                    if not sim.is_goal(path_cfgs[-1]):
                        centers = sim.goal_positions.astype(np.float32)
                        if sim.motion_valid(path_cfgs[-1], centers):
                            path_cfgs.append(centers)
                    sim.visualize_paths(path_cfgs)

    # Save trial-level CSV
    import csv
    csv_path = os.path.join(args.outdir, 'trials_q4.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_trial[0].keys()))
        writer.writeheader()
        for row in per_trial:
            writer.writerow(row)

    # Aggregate stats per (K, level)
    summary = {}
    levels = sorted(set(x['level'] for x in per_trial))
    k_groups = sorted(set(x['K'] for x in per_trial))
    for K in k_groups:
        summary[K] = {}
        for L in levels:
            rows = [r for r in per_trial if r['level'] == L and r['K'] == K]
            succ = [r['success'] for r in rows]
            times = [r['time_sec'] for r in rows if r['success'] == 1]
            pathlens = [r['path_len_joint'] for r in rows if r['success'] == 1 and not math.isnan(r['path_len_joint'])]

            def mc(samples):
                if len(samples) == 0:
                    return (float('nan'), (float('nan'), float('nan')))
                m = float(np.mean(samples))
                s = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
                hw = 1.96 * s / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
                return m, (m - hw, m + hw)

            sr_m, (sr_lo, sr_hi) = mc(succ)
            tm_m, (tm_lo, tm_hi) = mc(times) if len(times) > 0 else (float('nan'), (float('nan'), float('nan')))
            pl_m, (pl_lo, pl_hi) = mc(pathlens) if len(pathlens) > 0 else (float('nan'), (float('nan'), float('nan')))

            summary[K][L] = {
                'n_trials': len(rows),
                'success_rate_mean': sr_m,
                'success_rate_CI95': [sr_lo, sr_hi],
                'time_sec_mean_successes': tm_m,
                'time_sec_CI95_successes': [tm_lo, tm_hi],
                'path_len_mean_successes': pl_m,
                'path_len_CI95_successes': [pl_lo, pl_hi],
            }

    json_path = os.path.join(args.outdir, 'summary_q4.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote per-trial CSV -> {csv_path}")
    print(f"Wrote summary JSON -> {json_path}")

if __name__ == "__main__":
    main()
