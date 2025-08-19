#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

# Import the provided simulator
from multi_drone import MultiDrone


def set_global_seed(seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


@dataclass
class Node:
    q: np.ndarray  # shape (K, 3)
    parent: Optional[int]  # index of parent node in the tree or None for root


class Tree:
    """Simple tree for BiRRT-Connect storing nodes and linear nearest search."""

    def __init__(self, root: np.ndarray):
        self.nodes: List[Node] = [Node(q=root.copy(), parent=None)]

    def add(self, q: np.ndarray, parent_idx: int) -> int:
        self.nodes.append(Node(q=q.copy(), parent=parent_idx))
        return len(self.nodes) - 1

    def nearest(self, q: np.ndarray) -> int:
        """Return index of nearest node (Euclidean in R^{3K})."""
        qf = q.reshape(-1)
        best = 0
        best_d2 = float("inf")
        for i, node in enumerate(self.nodes):
            d2 = float(np.sum((node.q.reshape(-1) - qf) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = i
        return best

    def trace_path_to_root(self, idx: int) -> List[np.ndarray]:
        """Return list of configurations from root to idx (inclusive)."""
        out: List[np.ndarray] = []
        cur: Optional[int] = idx
        while cur is not None:
            out.append(self.nodes[cur].q.copy())
            cur = self.nodes[cur].parent
        out.reverse()
        return out


def uniform_sample_in_bounds(bounds: np.ndarray, K: int) -> np.ndarray:
    """Sample one configuration q ~ Uniform(bounds)^K.

    bounds: (3,2) array [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    returns: (K,3) array
    """
    low = bounds[:, 0]
    high = bounds[:, 1]
    q = np.random.uniform(low, high, size=(K, 3)).astype(np.float32)
    return q


def steer(q_from: np.ndarray, q_to: np.ndarray, delta_pos: float) -> np.ndarray:
    """Move from q_from towards q_to by at most delta_pos per drone."""
    v = q_to - q_from
    K = q_from.shape[0]
    out = q_from.copy().astype(np.float32)
    for i in range(K):
        norm = float(np.linalg.norm(v[i]))
        if norm <= 1e-9:
            out[i] = q_from[i]
        else:
            step = min(delta_pos, norm)
            out[i] = q_from[i] + (step / norm) * v[i]
    return out


def equal_config(q1: np.ndarray, q2: np.ndarray, tol: float = 1e-6) -> bool:
    return float(np.linalg.norm(q1.reshape(-1) - q2.reshape(-1))) <= tol


def connect(sim: MultiDrone, T: Tree, q_target: np.ndarray, delta_pos: float) -> Tuple[str, int]:
    """Greedily extend T towards q_target using 'steer' until trapped or reached."""
    while True:
        near_idx = T.nearest(q_target)
        q_near = T.nodes[near_idx].q
        q_new = steer(q_near, q_target, delta_pos)

        if not sim.motion_valid(q_near, q_new):
            return "Trapped", near_idx

        new_idx = T.add(q_new, parent_idx=near_idx)

        # If we reached or can directly connect to target, finish
        if equal_config(q_new, q_target) or sim.motion_valid(q_new, q_target):
            return "Reached", new_idx


def shortcut_smoothing(sim: MultiDrone, path: List[np.ndarray], trials: int = 200) -> List[np.ndarray]:
    if len(path) <= 2 or trials <= 0:
        return path
    path = list(path)
    for _ in range(trials):
        if len(path) <= 2:
            break
        a = random.randint(0, len(path) - 3)
        b = random.randint(a + 2, len(path) - 1)
        qa, qb = path[a], path[b]
        if sim.motion_valid(qa, qb):
            path = path[: a + 1] + path[b:]
    return path


def birrt_connect(sim: MultiDrone,
                  q_start: np.ndarray,
                  q_goal: np.ndarray,
                  time_limit: float = 120.0,
                  max_iters: int = 100000,
                  delta_pos: float = 0.75,
                  goal_bias: float = 0.1,
                  smoothing_trials: int = 200,
                  rng_seed: Optional[int] = None) -> Optional[List[np.ndarray]]:
    """Centralized BiRRT-Connect in R^{3K}."""
    assert q_start.shape == q_goal.shape and q_start.ndim == 2, "q_start/q_goal must be (K,3)"
    K = q_start.shape[0]

    set_global_seed(rng_seed)

    if not sim.is_valid(q_start):
        raise ValueError("Start configuration is invalid.")
    if not sim.is_valid(q_goal):
        raise ValueError("Goal configuration is invalid. Consider using goal centers that are valid.")

    T_a = Tree(root=q_start)
    T_b = Tree(root=q_goal)

    t0 = time.time()
    it = 0

    while it < max_iters and (time.time() - t0) < time_limit:
        it += 1

        # ----- sample with goal bias -----
        if random.random() < goal_bias:
            q_rand = q_goal
        else:
            # rejection sample
            for _ in range(5000):
                q_rand = uniform_sample_in_bounds(sim._bounds, K)  # sim bounds are (3,2)
                if sim.is_valid(q_rand):
                    break
            else:
                continue

        # ----- extend T_a toward q_rand -----
        near_idx = T_a.nearest(q_rand)
        q_near = T_a.nodes[near_idx].q
        q_new = steer(q_near, q_rand, delta_pos)

        if sim.motion_valid(q_near, q_new):
            new_idx = T_a.add(q_new, parent_idx=near_idx)

            # ----- greedily connect T_b to q_new -----
            status, meet_idx = connect(sim, T_b, q_new, delta_pos)
            if status == "Reached":
                path_a = T_a.trace_path_to_root(new_idx)      # start -> q_new
                path_b = T_b.trace_path_to_root(meet_idx)     # goal  -> q_new
                path_b.reverse()                               # q_new -> goal
                path = path_a + path_b[1:]                    # avoid duplicate q_new

                path = shortcut_smoothing(sim, path, trials=smoothing_trials)

                if not sim.is_goal(path[-1]):
                    if sim.motion_valid(path[-1], q_goal):
                        path.append(q_goal.copy())
                return path

        # Swap trees
        T_a, T_b = T_b, T_a

    return None


def build_goal_from_centers(sim: MultiDrone) -> np.ndarray:
    """Construct a goal configuration by using the goal sphere centers."""
    return sim.goal_positions.astype(np.float32)


def read_num_drones_from_yaml(env_path: str) -> int:
    with open(env_path, "r") as f:
        cfg = yaml.safe_load(f)
    init_cfg = cfg.get("initial_configuration", None)
    if not init_cfg or not isinstance(init_cfg, list):
        raise ValueError("environment YAML missing a valid 'initial_configuration'.")
    return len(init_cfg)


def main():
    parser = argparse.ArgumentParser(description="Centralized BiRRT-Connect for MultiDrone")
    parser.add_argument("--env", type=str, default="environment.yaml",
                        help="Path to environment YAML file.")
    parser.add_argument("--num-drones", type=int, default=None,
                        help="Number of drones K. If omitted, inferred from YAML initial_configuration.")
    parser.add_argument("--time-limit", type=float, default=110.0,
                        help="Planning time limit in seconds (<=120 for the demo).")
    parser.add_argument("--delta-pos", type=float, default=0.75,
                        help="Per-drone steering step (meters).")
    parser.add_argument("--goal-bias", type=float, default=0.1,
                        help="Goal bias probability.")
    parser.add_argument("--max-iters", type=int, default=100000,
                        help="Maximum RRT iterations.")
    parser.add_argument("--smooth-trials", type=int, default=200,
                        help="Shortcut smoothing trials.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization (useful for headless runs).")
    parser.add_argument("--save-path", type=str, default=None,
                        help="If set, save the path npy to this file.")
    args = parser.parse_args()

    # Infer K if not provided
    if args.num_drones is None:
        K = read_num_drones_from_yaml(args.env)
    else:
        K = args.num_drones

    # Initialize simulator
    sim = MultiDrone(num_drones=K, environment_file=args.env)

    q_start = sim.initial_configuration.astype(np.float32)
    q_goal = build_goal_from_centers(sim)               # goal sphere centers

    path = birrt_connect(sim,
                         q_start=q_start,
                         q_goal=q_goal,
                         time_limit=args.time_limit,
                         max_iters=args.max_iters,
                         delta_pos=args.delta_pos,
                         goal_bias=args.goal_bias,
                         smoothing_trials=args.smooth_trials,
                         rng_seed=args.seed)

    if path is None:
        print("Planning FAILED within the given limits.")
        return

    if not sim.is_goal(path[-1]):
        print("Warning: final waypoint not within goal spheres; attempting to append goal center.")
        if sim.motion_valid(path[-1], q_goal):
            path.append(q_goal.copy())

    print(f"Planning SUCCEEDED. Waypoints: {len(path)}")

    if args.save_path:
        np.save(args.save_path, np.stack(path, axis=0))
        print(f"Saved path to: {args.save_path}")

    if not args.no_viz:
        try:
            sim.visualize_paths(path)
        except Exception as e:
            print(f"Visualization failed or was closed: {e}")


if __name__ == "__main__":
    main()
