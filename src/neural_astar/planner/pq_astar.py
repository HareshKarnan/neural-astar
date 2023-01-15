"""Standard A* search with priority queue
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import numpy as np
import torch
from pqdict import pqdict

from .differentiable_astar import AstarOutput


def get_neighbor_indices(idx: int, H: int, W: int) -> np.array:
    """Get neighbor indices"""

    neighbor_indices = []
    if idx % W - 1 >= 0:
        neighbor_indices.append(idx - 1)
    if idx % W + 1 < W:
        neighbor_indices.append(idx + 1)
    if idx // W - 1 >= 0:
        neighbor_indices.append(idx - W)
    if idx // W + 1 < H:
        neighbor_indices.append(idx + W)
    if (idx % W - 1 >= 0) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W - 1)
    if (idx % W + 1 < W) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W + 1)
    if (idx % W - 1 >= 0) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W - 1)
    if (idx % W + 1 < W) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W + 1)

    return np.array(neighbor_indices)


def compute_chebyshev_distance(idx: int, goal_idx: int, W: int) -> float:
    """Compute chebyshev heuristic"""

    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    dxdy = np.abs(loc - goal_loc)
    h = dxdy.sum() - dxdy.min()
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return h + 0.001 * euc

def compute_manhattan_distance(idx, goal_idx, W) -> float:
    """compute manhattan heuristic"""
    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    dxdy = np.abs(loc - goal_loc)
    h = dxdy.sum()
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return h + 0.001 * euc

def compute_euclidean_distance(idx, goal_idx, W):
    """compute euclidean heuristic"""
    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return euc

def get_history(close_list: list, H: int, W: int) -> np.array:
    """Get search history"""

    history = np.array([[idx % W, idx // W] for idx in close_list.keys()])
    history_map = np.zeros((H, W))
    history_map[history[:, 1], history[:, 0]] = 1

    return history_map


def backtrack(parent_list: list, goal_idx: int, H: int, W: int) -> np.array:
    """Backtrack to obtain path"""

    current_idx = goal_idx
    path = []
    while current_idx != None:
        path.append([current_idx % W, current_idx // W])
        current_idx = parent_list[current_idx]
    path = np.array(path)
    path_map = np.zeros((H, W))
    path_map[path[:, 1], path[:, 0]] = 1

    return path_map


def pq_astar(
    pred_costs: np.array,
    start_maps: np.array,
    goal_maps: np.array,
    map_designs: np.array,
    store_intermediate_results: bool = False,
    g_ratio: float = 0.5,
) -> AstarOutput:
    """Perform standard A* on a batch of problems"""

    assert (
        store_intermediate_results == False
    ), "store_intermediate_results = True is currently supported only for differentiable A*"

    pred_costs_np = pred_costs.detach().numpy()
    start_maps_np = start_maps.detach().numpy()
    goal_maps_np = goal_maps.detach().numpy()
    map_designs_np = map_designs.detach().numpy()
    histories = np.zeros_like(goal_maps_np)
    path_maps = np.zeros_like(goal_maps_np)
    for n in range(len(pred_costs)):
        histories[n, 0], path_maps[n, 0] = solve_single(
            pred_costs_np[n, 0],
            start_maps_np[n, 0],
            goal_maps_np[n, 0],
            map_designs_np[n, 0],
            g_ratio,
        )

    return AstarOutput(torch.tensor(histories), torch.tensor(path_maps))

def solve_single(
    pred_cost: np.array,
    start_map: np.array,
    goal_map: np.array,
    map_design: np.array,
    g_ratio: float = 0.5,
) -> list:
    """Solve a single problem"""
    
    H, W = map_design.shape
    start_idx = np.argwhere(start_map.flatten()).item()
    goal_idx = np.argwhere(goal_map.flatten()).item()
    map_design_vct = map_design.flatten()
    pred_cost_vct = pred_cost.flatten()
    # normalize the pred_cost for each cell to be in [0, max_chebyshev] where max_chebyshev is the maximum distance from start_idx
    # max_chebyshev = compute_chebyshev_distance(start_idx, goal_idx, W)
    # pred_cost_vct = pred_cost_vct * np.sqrt(max_chebyshev)

    # pred_cost_vct is between 0-1
    # for A star to work, we need to balance the cost of the heuristic and the cost of the path
    # we do this by normalizing the pred_cost_vct based on the euclidean heuristic's cost
    # pred_cost_vct = pred_cost_vct * compute_euclidean_distance(start_idx, goal_idx, W)
    
    open_list = pqdict()
    close_list = pqdict()
    open_list.additem(start_idx, 0)
    parent_list = dict()
    parent_list[start_idx] = None

    num_steps = 0
    while goal_idx not in close_list:
        if len(open_list) == 0:
            print("goal not found")
            return np.zeros_like(goal_map), np.zeros_like(goal_map)
        num_steps += 1
        v_idx, v_cost = open_list.popitem()
        close_list.additem(v_idx, v_cost)
        for n_idx in get_neighbor_indices(v_idx, H, W):
            # f_new = (
            #     v_cost
            #     - (1 - g_ratio) * compute_manhattan_distance(v_idx, goal_idx, W)
            #     + g_ratio * pred_cost_vct[n_idx]
            #     + (1 - g_ratio) * compute_manhattan_distance(n_idx, goal_idx, W)
            # )
            f_new = (
                v_cost - compute_chebyshev_distance(v_idx, goal_idx, W)
                + pred_cost_vct[n_idx]
                + compute_chebyshev_distance(n_idx, goal_idx, W)
            )

            # conditions for the nodes not yet in the open list nor closed list
            cond = (n_idx not in open_list) & (n_idx not in close_list)

            # condition for the nodes already in the open list but with larger f value
            if n_idx in open_list:
                cond = cond | (open_list[n_idx] > f_new)

            if cond:
                try:
                    open_list.additem(n_idx, f_new)
                except:
                    open_list[n_idx] = f_new
                parent_list[n_idx] = v_idx

    history_map = get_history(close_list, H, W)
    path_map = backtrack(parent_list, goal_idx, H, W)
    return history_map, path_map