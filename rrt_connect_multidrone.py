#!/usr/bin/env python3
"""
RRT-Connect for centralized multi-drone motion planning (Part B, Q3).
"""
import argparse
import time
import random
import numpy as np
from typing import List, Optional, Tuple
import yaml
from multi_drone import MultiDrone

def read_num_drones_from_yaml(env_path: str) -> int:
    with open(env_path, "r") as f:
        cfg = yaml.safe_load(f)
    init = cfg.get("initial_configuration", None)
    if not init or not isinstance(init, list):
        raise ValueError("environment.yaml missing valid 'initial_configuration' list")
    return len(init)

def flatten(cfg: np.ndarray) -> np.ndarray:
    return cfg.reshape(-1)

def unflatten(vec: np.ndarray, N: int) -> np.ndarray:
    return vec.reshape(N, 3)

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

class Tree:
    def __init__(self, root: np.ndarray):
        self.nodes: List[np.ndarray] = [root.copy()]
        self.parents: List[int] = [-1]
    def add(self, node: np.ndarray, parent_idx: int) -> int:
        self.nodes.append(node.copy())
        self.parents.append(parent_idx)
        return len(self.nodes) - 1
    def nearest(self, q: np.ndarray) -> int:
        dmin = float("inf"); imin = 0
        for i, n in enumerate(self.nodes):
            d = dist(n, q)
            if d < dmin: dmin, imin = d, i
        return imin
    def path_to_root(self, idx: int) -> List[np.ndarray]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx]); idx = self.parents[idx]
        out.reverse(); return out

def sample_in_bounds(bounds: np.ndarray, goal: np.ndarray, goal_bias: float, rng: random.Random) -> np.ndarray:
    if rng.random() < goal_bias:
        return goal.copy()
    N = goal.size // 3
    xyz = np.empty((N, 3), dtype=np.float32)
    for i in range(N):
        for j in range(3):
            lo, hi = float(bounds[j, 0]), float(bounds[j, 1])
            xyz[i, j] = rng.uniform(lo, hi)
    return flatten(xyz)

def steer(q_from: np.ndarray, q_to: np.ndarray, step: float) -> np.ndarray:
    d = q_to - q_from; L = float(np.linalg.norm(d))
    if L <= step: return q_to.copy()
    return q_from + (step / L) * d

def extend(sim: MultiDrone, tree: Tree, q_target: np.ndarray, step: float, N: int):
    idx_near = tree.nearest(q_target); q_near = tree.nodes[idx_near]
    q_new = steer(q_near, q_target, step)
    if not sim.motion_valid(unflatten(q_near, N), unflatten(q_new, N)):
        return "Trapped", None, q_near
    new_idx = tree.add(q_new, idx_near)
    if np.allclose(q_new, q_target, atol=1e-6):
        return "Reached", new_idx, q_new
    else:
        return "Advanced", new_idx, q_new

def connect(sim: MultiDrone, tree: Tree, q_target: np.ndarray, step: float, N: int):
    status = "Advanced"; last_idx = -1; last_q = tree.nodes[-1]
    while status == "Advanced":
        status, last_idx, last_q = extend(sim, tree, q_target, step, N)
        if status == "Trapped":
            return "Trapped", -1, last_q
    return status, last_idx, last_q

def reconstruct_bidirectional_path(a: Tree, idx_a: int, b: Tree, idx_b: int) -> List[np.ndarray]:
    pa = a.path_to_root(idx_a); pb = b.path_to_root(idx_b); pb.reverse()
    return pa + pb

def postprocess_shortcut(sim: MultiDrone, path: List[np.ndarray], N: int, rng: random.Random, iters: int = 200) -> List[np.ndarray]:
    if len(path) <= 2: return path
    path_cfgs = [unflatten(p, N) for p in path]
    for _ in range(iters):
        if len(path_cfgs) <= 2: break
        i = rng.randrange(0, len(path_cfgs) - 2)
        j = rng.randrange(i + 2, len(path_cfgs))
        if sim.motion_valid(path_cfgs[i], path_cfgs[j]):
            del path_cfgs[i + 1 : j]
    return [flatten(p) for p in path_cfgs]

def rrt_connect_plan(sim: MultiDrone, time_limit: float = 120.0, step: float = 1.0,
                     goal_bias: float = 0.2, seed: int = 0, max_nodes: int = 100000,
                     do_postprocess: bool = True):
    rng = random.Random(seed); N = sim.N
    q_start_cfg = sim.initial_configuration.astype(np.float32)
    q_goal_cfg_center = sim.goal_positions.astype(np.float32)
    if not sim.is_valid(q_start_cfg):
        raise RuntimeError("Initial configuration is invalid")
    q_start = flatten(q_start_cfg); q_goal = flatten(q_goal_cfg_center)
    start_time = time.time(); a = Tree(q_start); b = Tree(q_goal)
    bounds = sim._bounds.astype(np.float32)
    success = False; meet_a = meet_b = -1; it = 0
    if sim.is_goal(q_start_cfg):
        return True, [q_start, q_goal], {"iterations": 0, "nodes": 2, "time": 0.0}
    while time.time() - start_time < time_limit and (len(a.nodes) + len(b.nodes)) < max_nodes:
        it += 1; q_rand = sample_in_bounds(bounds, q_goal, goal_bias, rng)
        status, idx_new_a, q_new_a = extend(sim, a, q_rand, step, N)
        if status != "Trapped":
            status2, idx_new_b, q_new_b = connect(sim, b, q_new_a, step, N)
            if status2 == "Reached":
                meet_a, meet_b = idx_new_a, idx_new_b; success = True; break
        a, b = b, a
    elapsed = time.time() - start_time
    stats = {"iterations": it, "nodes": len(a.nodes) + len(b.nodes), "time": elapsed, "success": success}
    if not success: return False, [], stats
    path_flat = reconstruct_bidirectional_path(a, meet_a, b, meet_b)
    if do_postprocess: path_flat = postprocess_shortcut(sim, path_flat, N, rng=rng, iters=200)
    return True, path_flat, stats

def main():
    parser = argparse.ArgumentParser(description="RRT-Connect for MultiDrone (Part B, Q3)")
    parser.add_argument("--env", type=str, default="environment.yaml")
    parser.add_argument("--num-drones", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--goal-bias", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()
    K = args.num_drones if args.num_drones is not None else read_num_drones_from_yaml(args.env)
    sim = MultiDrone(num_drones=K, environment_file=args.env)
    ok, path_flat, stats = rrt_connect_plan(sim, time_limit=args.time_limit, step=args.step,
                                            goal_bias=args.goal_bias, seed=args.seed,
                                            do_postprocess=(not args.no_post))
    print(f"success={ok} iterations={stats.get('iterations')} nodes={stats.get('nodes')} time={stats.get('time'):.3f}s")
    if ok and args.viz:
        path_cfgs = [p.reshape(K, 3).astype(np.float32) for p in path_flat]
        if not sim.is_goal(path_cfgs[-1]):
            centers = sim.goal_positions.astype(np.float32)
            if sim.motion_valid(path_cfgs[-1], centers):
                path_cfgs.append(centers)
        sim.visualize_paths(path_cfgs)

if __name__ == "__main__":
    main()
