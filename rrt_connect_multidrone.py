#!/usr/bin/env python3
"""
RRT-Connect for centralized multi-drone motion planning (Part B, Q3).

Implements a sampling-based planner using the provided MultiDrone environment:
 - Configuration validity: sim.is_valid(config)  # shape (N, 3)
 - Motion validity: sim.motion_valid(q0, q1)
 - Goal detection: sim.is_goal(config)
 - Visualization: sim.visualize_paths(path_list)

Usage:
    python rrt_connect_multidrone.py --env environment.yaml [--time-limit 120] [--step 1.0] [--goal-bias 0.2] [--seed 0] [--viz]

Notes:
 - Centralized C-space of dimension 3K (K drones), each node stores all drones' positions.
 - RRT-Connect alternates growth from start and goal trees and attempts greedy connection.
 - Designed to comply with the 2-minute, single-threaded constraint.
"""
import argparse
import time
import math
import random
import numpy as np
from typing import List, Optional, Tuple
import yaml

from multi_drone import MultiDrone

# --------------------------- Utils ---------------------------

def read_num_drones_from_yaml(env_path: str) -> int:
    with open(env_path, "r") as f:
        cfg = yaml.safe_load(f)
    init = cfg.get("initial_configuration", None)
    if not init or not isinstance(init, list):
        raise ValueError("environment.yaml missing valid 'initial_configuration' list")
    return len(init)

def flatten(cfg: np.ndarray) -> np.ndarray:
    """(N,3) -> (3N,)"""
    return cfg.reshape(-1)

def unflatten(vec: np.ndarray, N: int) -> np.ndarray:
    """(3N,) -> (N,3)"""
    return vec.reshape(N, 3)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

# --------------------------- RRT-Connect Core ---------------------------

class Tree:
    def __init__(self, root: np.ndarray):
        self.nodes: List[np.ndarray] = [root.copy()]
        self.parents: List[int] = [-1]

    def add(self, node: np.ndarray, parent_idx: int) -> int:
        self.nodes.append(node.copy())
        self.parents.append(parent_idx)
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        # Linear scan (fast enough in practice for this assignment)
        dmin = float("inf")
        imin = 0
        for i, n in enumerate(self.nodes):
            d = dist(n, q)
            if d < dmin:
                dmin = d
                imin = i
        return imin

    def path_to_root(self, idx: int) -> List[np.ndarray]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parents[idx]
        out.reverse()
        return out

def sample_in_bounds(bounds: np.ndarray, goal: np.ndarray, goal_bias: float, rng: random.Random) -> np.ndarray:
    """
    bounds: (3,2), goal: (3N,), returns (3N,)
    Goal-biased sampling; when sampled, use goal center.
    """
    if rng.random() < goal_bias:
        return goal.copy()
    # Uniform in bounds per drone
    N = goal.size // 3
    xyz = np.empty((N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(3):
            lo, hi = float(bounds[j, 0]), float(bounds[j, 1])
            xyz[i, j] = rng.uniform(lo, hi)
    return flatten(xyz)

def steer(q_from: np.ndarray, q_to: np.ndarray, step: float) -> np.ndarray:
    """Move from q_from towards q_to by at most 'step' in Euclidean norm."""
    d = q_to - q_from
    L = float(np.linalg.norm(d))
    if L <= step:
        return q_to.copy()
    return q_from + (step / L) * d

def extend(sim: MultiDrone, tree: Tree, q_target: np.ndarray, step: float, N: int) -> Tuple[str, Optional[int], np.ndarray]:
    """
    Try to advance the tree by one step towards q_target.
    Returns (status, new_index, q_new):
        status in {"Trapped", "Advanced", "Reached"}
    """
    idx_near = tree.nearest(q_target)
    q_near = tree.nodes[idx_near]
    q_new = steer(q_near, q_target, step)

    q_near_cfg = unflatten(q_near, N)
    q_new_cfg = unflatten(q_new, N)

    # Check motion between nearest and new
    if not sim.motion_valid(q_near_cfg, q_new_cfg):
        return "Trapped", None, q_near

    new_idx = tree.add(q_new, idx_near)
    if np.allclose(q_new, q_target, atol=1e-6):
        return "Reached", new_idx, q_new
    else:
        return "Advanced", new_idx, q_new

def connect(sim: MultiDrone, tree: Tree, q_target: np.ndarray, step: float, N: int) -> Tuple[str, int, np.ndarray]:
    """
    Greedily extend towards q_target until trapped or reached.
    Returns (status, last_idx, last_q)
    """
    status = "Advanced"
    last_idx = -1
    last_q = tree.nodes[-1]
    while status == "Advanced":
        status, last_idx, last_q = extend(sim, tree, q_target, step, N)
        if status == "Trapped":
            return "Trapped", -1, last_q
    return status, last_idx, last_q

def reconstruct_bidirectional_path(a: Tree, idx_a: int, b: Tree, idx_b: int) -> List[np.ndarray]:
    """
    Path from a.root -> a[idx_a] + b[idx_b] -> b.root (reverse b segment).
    Returns a list of flattened configs.
    """
    pa = a.path_to_root(idx_a)
    pb = b.path_to_root(idx_b)
    pb.reverse()  # from b[idx_b] down to root
    return pa + pb

def postprocess_shortcut(sim: MultiDrone, path: List[np.ndarray], N: int, rng: random.Random, iters: int = 200) -> List[np.ndarray]:
    """Simple shortcutting to reduce waypoint count while preserving validity."""
    if len(path) <= 2:
        return path
    path_cfgs = [unflatten(p, N) for p in path]
    for _ in range(iters):
        if len(path_cfgs) <= 2:
            break
        i = rng.randrange(0, len(path_cfgs) - 2)
        j = rng.randrange(i + 2, len(path_cfgs))
        if sim.motion_valid(path_cfgs[i], path_cfgs[j]):
            # remove intermediates
            del path_cfgs[i + 1 : j]
    return [flatten(p) for p in path_cfgs]

# --------------------------- Planner ---------------------------

def rrt_connect_plan(
    sim: MultiDrone,
    time_limit: float = 120.0,
    step: float = 1.0,
    goal_bias: float = 0.2,
    seed: int = 0,
    max_nodes: int = 100000,
    do_postprocess: bool = True,
) -> Tuple[bool, List[np.ndarray], dict]:
    """
    Returns (success, path_flat_list, stats)
    path_flat_list is a list of flattened (3N,) vectors including start and goal.
    """
    rng = random.Random(seed)
    N = sim.N

    q_start_cfg = sim.initial_configuration.astype(np.float32)
    q_goal_cfg_center = sim.goal_positions.astype(np.float32)  # use goal centers
    if not sim.is_valid(q_start_cfg):
        raise RuntimeError("Initial configuration is invalid")
    # It's okay if goal centers are invalid; RRT-Connect will try to reach any point in the goal regions.

    q_start = flatten(q_start_cfg)
    q_goal = flatten(q_goal_cfg_center)

    start_time = time.time()
    a = Tree(q_start)
    b = Tree(q_goal)

    bounds = sim._bounds.astype(np.float32)  # Using provided environment bounds
    success = False
    meet_a = meet_b = -1
    it = 0

    # Early exit if already at goal
    if sim.is_goal(q_start_cfg):
        return True, [q_start, q_goal], {"iterations": 0, "nodes": 2, "time": 0.0}

    while time.time() - start_time < time_limit and (len(a.nodes) + len(b.nodes)) < max_nodes:
        it += 1
        q_rand = sample_in_bounds(bounds, q_goal, goal_bias, rng)

        # Grow a towards q_rand
        status, idx_new_a, q_new_a = extend(sim, a, q_rand, step, N)
        if status != "Trapped":
            # Try to connect b towards the newly added node
            status2, idx_new_b, q_new_b = connect(sim, b, q_new_a, step, N)
            if status2 == "Reached":
                meet_a, meet_b = idx_new_a, idx_new_b
                success = True
                break

        # Swap trees for the next iteration
        a, b = b, a

    elapsed = time.time() - start_time

    stats = {
        "iterations": it,
        "nodes": len(a.nodes) + len(b.nodes),
        "time": elapsed,
        "success": success,
    }

    if not success:
        return False, [], stats

    # Reconstruct path; 'a' and 'b' may have been swapped odd times; resolve using last step data.
    # We know q_new_a belongs to the tree we called 'a' at connection time.
    path_flat = reconstruct_bidirectional_path(a, meet_a, b, meet_b)

    # Post-process (shortcut) for fewer waypoints
    if do_postprocess:
        path_flat = postprocess_shortcut(sim, path_flat, N, rng=rng, iters=200)

    return True, path_flat, stats

# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="RRT-Connect for MultiDrone (Part B, Q3)")
    parser.add_argument("--env", type=str, default="environment.yaml", help="Path to environment YAML")
    parser.add_argument("--num-drones", type=int, default=None, help="Number of drones K; default: read from YAML initial_configuration length")
    parser.add_argument("--time-limit", type=float, default=120.0, help="Time budget in seconds")
    parser.add_argument("--step", type=float, default=1.0, help="Step size in configuration L2 norm")
    parser.add_argument("--goal-bias", type=float, default=0.2, help="Probability of sampling the goal")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--no-post", action="store_true", help="Disable path shortcutting")
    parser.add_argument("--viz", action="store_true", help="Visualize the resulting path(s)")
    args = parser.parse_args()

    # Determine K (num_drones)
    K = args.num_drones if args.num_drones is not None else read_num_drones_from_yaml(args.env)

    # Initialize simulator
    sim = MultiDrone(num_drones=K, environment_file=args.env)

    ok, path_flat, stats = rrt_connect_plan(
        sim,
        time_limit=args.time_limit,
        step=args.step,
        goal_bias=args.goal_bias,
        seed=args.seed,
        do_postprocess=(not args.no_post),
    )

    print(f"success={ok} iterations={stats.get('iterations')} nodes={stats.get('nodes')} time={stats.get('time'):.3f}s")

    if ok:
        # Convert to list of (N,3) ndarrays for visualization
        path_cfgs = [p.reshape(K, 3).astype(np.float32) for p in path_flat]

        # Sanity-check goal
        if not sim.is_goal(path_cfgs[-1]):
            # Try to push the final node to the goal centers if straight-line is valid
            centers = sim.goal_positions.astype(np.float32)
            if sim.motion_valid(path_cfgs[-1], centers):
                path_cfgs.append(centers)

        if args.viz:
            sim.visualize_paths(path_cfgs)

if __name__ == "__main__":
    main()
