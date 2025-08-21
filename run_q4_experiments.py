"""
Part B, Q4 runner
- Random obstacles can span almost floor→ceiling (prevents easy overflight).
- L4 narrow passage walls are full-height; gap width ~= (2*drone_radius + 0.05).
- Optional “ceiling slab” for levels >=3 to confine vertical motion.

"""

import argparse, os, time, math, json, random, sys
import numpy as np
import yaml
from typing import Dict, Any, Tuple, List


from multi_drone import MultiDrone
from rrt_connect_multidrone import rrt_connect_plan

# ---------------------- helpers ----------------------

def read_base(env_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(env_path, 'r') as f:
        data = yaml.safe_load(f)
    bounds = np.array([data['bounds']['x'], data['bounds']['y'], data['bounds']['z']], dtype=np.float32)
    init = np.array(data['initial_configuration'], dtype=np.float32)
    goals = np.array([g['position'] for g in data['goals']], dtype=np.float32)
    return bounds, init, goals

def clamp_to_bounds(arr: np.ndarray, bounds: np.ndarray, margin: float=0.5) -> np.ndarray:
    lo = bounds[:,0] + margin
    hi = bounds[:,1] - margin
    return np.minimum(hi, np.maximum(lo, arr))

def area_capacity(bounds: np.ndarray, sep: float) -> int:
    w = bounds[0,1] - bounds[0,0] - 1.0  # margins
    h = bounds[1,1] - bounds[1,0] - 1.0
    if w <= 0 or h <= 0 or sep <= 0: return 0
    cols = int(w // sep); rows = int(h // sep)
    return max(0, rows * cols)

def grid_layout(bounds: np.ndarray, K: int, sep: float, z_ref: float) -> np.ndarray:
    """Return Kx3 grid inside bounds using spacing 'sep' (square packing)."""
    w = bounds[0,1] - bounds[0,0] - 1.0
    h = bounds[1,1] - bounds[1,0] - 1.0
    cols = max(1, int(w // sep))
    rows = max(1, int(h // sep))
    if rows * cols < K:
        cols = max(cols, int(math.ceil(math.sqrt(K))))
        rows = max(rows, int(math.ceil(K / cols)))
    x0 = (bounds[0,0] + bounds[0,1]) * 0.5 - (cols-1) * sep * 0.5
    y0 = (bounds[1,0] + bounds[1,1]) * 0.5 - (rows-1) * sep * 0.5
    pts = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= K: break
            pts.append([x0 + c*sep, y0 + r*sep, z_ref])
            idx += 1
        if idx >= K: break
    return clamp_to_bounds(np.array(pts, dtype=np.float32), bounds, margin=0.5)

def min_pairwise_dist(arr: np.ndarray) -> float:
    if len(arr) < 2: return float('inf')
    D = np.linalg.norm(arr[:,None,:]-arr[None,:,:], axis=-1)
    mask = np.triu(np.ones((len(arr),len(arr))),1).astype(bool)
    return float(np.min(D[mask]))

def randf(lo, hi, rng) -> float:
    return float(rng.uniform(lo, hi))

def make_box(position, size, rotation=(0,0,0), color='red'):
    return {'type':'box', 'position': [float(x) for x in position], 'size': [float(x) for x in size], 'rotation': list(rotation), 'color': color}

def make_sphere(position, radius, rotation=(0,0,0), color='red'):
    return {'type':'sphere', 'position': [float(x) for x in position], 'radius': float(radius), 'rotation': list(rotation), 'color': color}

def make_cylinder(p1, p2, radius, rotation=(0,0,0), color='red'):
    return {'type':'cylinder', 'endpoints': [[float(x) for x in p1], [float(x) for x in p2]], 'radius': float(radius), 'rotation': list(rotation), 'color': color}

def obstacle_clear_from_points(ob, points: np.ndarray, clearance: float, bounds: np.ndarray) -> bool:
    """Conservative clearance check: keep a buffer 'clearance' (meters) from every point.""" # the automatic check that is performed.
    pts = points
    if ob['type'] == 'sphere':
        c = np.array(ob['position'], dtype=np.float32)
        rad = float(ob['radius'])
        d = np.min(np.linalg.norm(pts - c[None,:], axis=-1))
        return d >= (rad + clearance)
    elif ob['type'] == 'box':
        c = np.array(ob['position'], dtype=np.float32)
        half = 0.5 * np.array(ob['size'], dtype=np.float32)
        half_exp = half + clearance
        diff = np.abs(pts - c[None,:]) - half_exp[None,:]
        diff_clip = np.maximum(diff, 0.0)
        d = np.min(np.linalg.norm(diff_clip, axis=-1))
        return d > 0.0
    elif ob['type'] == 'cylinder':
        p1 = np.array(ob['endpoints'][0], dtype=np.float32)
        p2 = np.array(ob['endpoints'][1], dtype=np.float32)
        r = float(ob['radius']) + clearance
        axis = p2 - p1
        axis_xy = axis.copy(); axis_xy[2] = 0.0
        for p in pts:
            if not (min(p1[2],p2[2]) - clearance <= p[2] <= max(p1[2],p2[2]) + clearance):
                continue
            v = p[:2] - p1[:2]
            if np.linalg.norm(axis_xy[:2]) < 1e-6:
                dxy = np.linalg.norm(v)
            else:
                t = np.dot(v, axis_xy[:2]) / (np.dot(axis_xy[:2], axis_xy[:2]))
                t = np.clip(t, 0.0, 1.0)
                closest = p1[:2] + t*axis_xy[:2]
                dxy = np.linalg.norm(p[:2] - closest)
            if dxy < r:
                return False
        return True
    else:
        return True

def generate_env(bounds: np.ndarray, init: np.ndarray, goals: np.ndarray, level: int, seed: int, clearance: float, debug: int, rng: random.Random) -> Dict[str, Any]:
    """Create an environment; obstacles respect a 'clearance' from starts/goals and can be full-height.""" # as per the question demand the fucntion has been executed
    env = {
        'bounds': {'x': bounds[0].tolist(), 'y': bounds[1].tolist(), 'z': bounds[2].tolist()},
        'initial_configuration': init.tolist(),
        'obstacles': [],
        'goals': [{'position': g.tolist(), 'radius': 1.0} for g in goals],
    }
    # Allow tall obstacles (avoid trivial overflight) # altered the osbtacles again for including tall objects to see the path being used.
    ZR = (bounds[2,0]+0.5, bounds[2,1]-0.5)
    XR = (bounds[0,0]+2.0, bounds[0,1]-2.0); YR = (bounds[1,0]+2.0, bounds[1,1]-2.0)

    def try_add_obstacle(ob):
        if obstacle_clear_from_points(ob, init, clearance, bounds) and obstacle_clear_from_points(ob, goals, clearance, bounds):
            env['obstacles'].append(ob); return True
        return False

    def add_random_obstacle():
        typ = rng.choice(['box','sphere','cylinder'])
        if typ == 'box':
            pos = np.array([randf(*XR, rng), randf(*YR, rng), randf(*ZR, rng)], dtype=np.float32)
            # z-size can be large (up to almost ceiling)   # later amended by adding a roofing to ensure the results are not false if path went out of bounds.
            z_span = randf(1.0, max(1.0, (bounds[2,1]-bounds[2,0]) - 1.0), rng)
            size = np.array([randf(1.5,4.5,rng), randf(1.5,4.5,rng), z_span], dtype=np.float32)
            try_add_obstacle(make_box(pos, size))
        elif typ == 'sphere':
            pos = np.array([randf(*XR, rng), randf(*YR, rng), randf(*ZR, rng)], dtype=np.float32)
            radius = randf(1.0, 2.5, rng)
            try_add_obstacle(make_sphere(pos, radius))
        else:
            # vertical cylinder spanning almost full height
            p1 = np.array([randf(*XR, rng), randf(*YR, rng), bounds[2,0]+0.5], dtype=np.float32)
            p2 = p1.copy(); p2[2] = bounds[2,1]-0.5
            radius = randf(1.0, 2.5, rng)
            try_add_obstacle(make_cylinder(p1, p2, radius))

    if level == 0:
        pass
    elif level == 1:
        for _ in range(rng.randint(2,3)): add_random_obstacle()
    elif level == 2:
        for _ in range(rng.randint(4,5)): add_random_obstacle()
    elif level == 3:
        for _ in range(rng.randint(6,8)): add_random_obstacle()
        # optional shallow ceiling slab to confine motion #the final amend made to ensure the traversal path is not out of bounds.
        x_len = (bounds[0,1]-bounds[0,0]) - 2.0
        y_len = (bounds[1,1]-bounds[1,0]) - 2.0
        z_mid = bounds[2,1]-0.7
        try_add_obstacle(make_box([ (bounds[0,0]+bounds[0,1])/2,
                                    (bounds[1,0]+bounds[1,1])/2,
                                    z_mid ],
                                  [x_len, y_len, 1.0], color='gray'))
    elif level == 4:
        # Two full-height walls with a narrow gap tied to drone size
        gap_width = clearance + 0.05  # ~= 2*drone_radius + 0.05
        x_mid = (bounds[0,0] + bounds[0,1]) * 0.5
        y_mid = (bounds[1,0] + bounds[1,1]) * 0.5
        z_mid = (bounds[2,0] + bounds[2,1]) * 0.5
        z_span = (bounds[2,1] - bounds[2,0]) - 1.0  # leave small margins
        left_len = (x_mid - gap_width*0.5) - (bounds[0,0]+1.0)
        if left_len > 0.5:
            left_center = np.array([bounds[0,0]+1.0 + left_len*0.5, y_mid, z_mid], dtype=np.float32)
            left_size   = np.array([left_len, 2.0, z_span], dtype=np.float32)
            try_add_obstacle(make_box(left_center, left_size, color='red'))
        right_len = (bounds[0,1]-1.0) - (x_mid + gap_width*0.5)
        if right_len > 0.5:
            right_center = np.array([x_mid + gap_width*0.5 + right_len*0.5, y_mid, z_mid], dtype=np.float32)
            right_size   = np.array([right_len, 2.0, z_span], dtype=np.float32)
            try_add_obstacle(make_box(right_center, right_size, color='red'))
        # extra clutter
        for _ in range(2): add_random_obstacle()
        # ceiling slab too
        x_len = (bounds[0,1]-bounds[0,0]) - 2.0
        y_len = (bounds[1,1]-bounds[1,0]) - 2.0
        z_mid2 = bounds[2,1]-0.7
        try_add_obstacle(make_box([ (bounds[0,0]+bounds[0,1])/2,
                                    (bounds[1,0]+bounds[1,1])/2,
                                    z_mid2 ],
                                  [x_len, y_len, 1.0], color='gray'))
    else:
        raise ValueError("Unsupported complexity level")

    if debug:
        print(f"[DBG] Generated env with {len(env['obstacles'])} obstacles, full-height enabled.")
    return env

def build_starts_goals(bounds: np.ndarray, base_init: np.ndarray, base_goals: np.ndarray, K: int, drone_radius: float, min_sep_req: float, debug: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (init Kx3, goals Kx3, used_min_sep). Adapts min_sep if area can't fit K."""
    z_ref_s = float(np.mean(base_init[:,2]))
    z_ref_g = float(np.mean(base_goals[:,2]))
    min_sep_floor = 2.0 * drone_radius
    sep = max(min_sep_req, min_sep_floor)

    # Adapt sep downwards (never below floor) until capacity suffices
    max_iter = 20
    for _ in range(max_iter):
        cap = area_capacity(bounds, sep)
        if cap >= K or sep <= min_sep_floor + 1e-6:
            break
        sep *= 0.9
    if debug:
        print(f"[DBG] K={K}: requested min_sep={min_sep_req:.3f}, floor={min_sep_floor:.3f}, used={sep:.3f}, capacity={area_capacity(bounds, sep)}")

    init = grid_layout(bounds, K, sep, z_ref_s)
    goals = grid_layout(bounds, K, sep, z_ref_g)

    # Final clamp
    init = clamp_to_bounds(init, bounds, margin=0.5)
    goals = clamp_to_bounds(goals, bounds, margin=0.5)

    return init.astype(np.float32), goals.astype(np.float32), float(sep)

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-env', type=str, default='environment.yaml')
    ap.add_argument('--levels', type=int, nargs='+', default=[0,1,2,3,4])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--k', type=int, default=None)
    ap.add_argument('--k-list', type=int, nargs='+', default=None)
    ap.add_argument('--k-range', type=int, nargs=2, default=None)
    ap.add_argument('--time-limit', type=float, default=120.0)
    ap.add_argument('--step', type=float, default=1.0)
    ap.add_argument('--goal-bias', type=float, default=0.25)
    ap.add_argument('--outdir', type=str, default='results_q4')
    ap.add_argument('--viz', type=int, default=0)
    ap.add_argument('--drone-radius', type=float, default=0.3)
    ap.add_argument('--min-sep', type=float, default=None)
    ap.add_argument('--debug', type=int, default=0)
    args = ap.parse_args()

    print(f"[INFO] Working dir: {os.getcwd()}")
    out_abs = os.path.abspath(args.outdir)
    os.makedirs(out_abs, exist_ok=True)
    print(f"[INFO] Using outdir: {out_abs}")

    # Load base
    with open(args.base_env, 'r') as f:
        base = yaml.safe_load(f)
    k_base = len(base['initial_configuration'])
    bounds, base_init, base_goals = read_base(args.base_env)

    # K sweep
    if args.k_list is not None: k_values = args.k_list
    elif args.k_range is not None: start, end = args.k_range; k_values = list(range(start, end+1))
    else: k_values = [args.k or k_base]

    min_sep_req = (2.0*args.drone_radius + 0.1) if args.min_sep is None else float(args.min_sep)
    clearance = max(0.5, args.drone_radius*2.0)  # keep obstacles at least this far from starts/goals

    per_trial = []
    import csv

    for K in k_values:
        for level in args.levels:
            for seed in args.seeds:
                rng = random.Random(seed + 12345*K + 100*level)

                # Build spaced starts/goals (grid-based)
                init, goals, used_sep = build_starts_goals(bounds, base_init, base_goals, K, args.drone_radius, min_sep_req, args.debug)

                # Try multiple environments until init is valid
                ok_init = False; env_path = None; tries = 0
                for attempt in range(20):
                    env = generate_env(bounds, init, goals, level, seed + 10000*attempt, clearance, args.debug, rng)
                    env_path = os.path.join(out_abs, f'env_K{K}_level{level}_seed{seed}_try{attempt}.yaml')
                    with open(env_path, 'w') as f:
                        yaml.safe_dump(env, f, sort_keys=False)
                    sim = MultiDrone(num_drones=K, environment_file=env_path)
                    tries += 1
                    valid = sim.is_valid(init)
                    mind = min_pairwise_dist(init)
                    if args.debug:
                        print(f"[DBG] K={K} L={level} seed={seed} try={attempt}: min-start-dist={mind:.3f} used-sep={used_sep:.3f} is_valid={valid} path={env_path}")
                    if valid:
                        ok_init = True
                        break
                if not ok_init:
                    print(f"[WARN] Skip trial: K={K} level={level} seed={seed} (no valid start after {tries} envs).")
                    continue

                # Run planner
                t0 = time.time()
                ok, path_flat, stats = rrt_connect_plan(
                    sim, time_limit=args.time_limit, step=args.step,
                    goal_bias=args.goal_bias, seed=seed, do_postprocess=True)
                elapsed = time.time() - t0

                path_len = (sum(float(np.linalg.norm(a-b)) for a,b in zip(path_flat[:-1], path_flat[1:]))
                            if ok and len(path_flat) >= 2 else float('nan'))

                per_trial.append({
                    'K': K, 'level': level, 'seed': seed, 'success': int(ok),
                    'time_sec': stats.get('time', elapsed), 'nodes': stats.get('nodes', None),
                    'iterations': stats.get('iterations', None), 'path_len_joint': path_len,
                })
                if args.debug:
                    print(f"[DBG] -> success={ok} time={stats.get('time', elapsed):.3f}s nodes={stats.get('nodes',-1)}")

    if len(per_trial)==0:
        print("[ERROR] No trials completed. Nothing to write. Check earlier warnings above.")
        sys.exit(2)

    csv_path = os.path.join(out_abs, 'trials_q4.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_trial[0].keys()))
        writer.writeheader()
        for row in per_trial:
            writer.writerow(row)
    print(f"[INFO] Wrote per-trial CSV -> {csv_path}")

    # Aggregate summary
    summary = {}
    levels = sorted(set(x['level'] for x in per_trial)); k_groups = sorted(set(x['K'] for x in per_trial))
    def mc(samples):
        if len(samples)==0: return (float('nan'), (float('nan'), float('nan')))
        m = float(np.mean(samples)); s = float(np.std(samples, ddof=1)) if len(samples)>1 else 0.0
        hw = 1.96*s/math.sqrt(len(samples)) if len(samples)>1 else 0.0
        return m, (m-hw, m+hw)
    for K in k_groups:
        summary[K] = {}
        for L in levels:
            rows = [r for r in per_trial if r['level']==L and r['K']==K]
            succ = [r['success'] for r in rows]
            times = [r['time_sec'] for r in rows if r['success']==1]
            plens = [r['path_len_joint'] for r in rows if r['success']==1 and not math.isnan(r['path_len_joint'])]
            sr,(srlo,srhi) = mc(succ)
            if len(times)>0: tm,(tmlo,tmhi) = mc(times)
            else: tm,tmlo,tmhi = float('nan'), float('nan'), float('nan')
            if len(plens)>0: pl,(pllo,plhi) = mc(plens)
            else: pl,pllo,plhi = float('nan'), float('nan'), float('nan')
            summary[K][L] = {
                'n_trials': len(rows),
                'success_rate_mean': sr, 'success_rate_CI95': [srlo, srhi],
                'time_sec_mean_successes': tm, 'time_sec_CI95_successes': [tmlo, tmhi],
                'path_len_mean_successes': pl, 'path_len_CI95_successes': [pllo, plhi],
            }

    json_path = os.path.join(out_abs, 'summary_q4.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote summary JSON -> {json_path}")

if __name__ == "__main__":
    main()
