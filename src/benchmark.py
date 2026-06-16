"""Benchmark chuẩn Solomon VRPTW — để đánh giá thuật toán theo đúng thông lệ học thuật.

Bộ Solomon (1987) là chuẩn vàng cho VRPTW: 6 lớp (C1/C2/R1/R2/RC1/RC2) × nhiều
instance, mỗi instance 100 khách. Khoảng cách & thời gian di chuyển = khoảng cách
Euclid (vận tốc = 1). Mục tiêu kinh điển: tối thiểu SỐ XE rồi TỔNG QUÃNG ĐƯỜNG.

Module này:
  - load_solomon(): đọc file chuẩn -> ma trận + ràng buộc.
  - DistanceModel: bộ tiêu chí cho ACOSolver để tối thiểu QUÃNG ĐƯỜNG (thay vì
    nhiên liệu PRP), nhờ vậy so sánh được với best-known của tài liệu.
  - run_on_instance(): chạy một thuật toán và trả về (số xe, quãng đường, ...).
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import ACOConfig
from src.aco_solver import ACOSolver, VehicleState
from src import baseline


# Best-known solutions (tham khảo, SINTEF/Solomon) cho bản 100 khách —
# (số xe, tổng quãng đường). Giá trị có thể chênh nhẹ giữa các nguồn.
BKS_100 = {
    "C101": (10, 828.94), "C201": (3, 591.56),
    "R101": (19, 1650.80), "R201": (4, 1252.37),
    "RC101": (14, 1696.95), "RC201": (4, 1406.94),
}


@dataclass
class BenchmarkInstance:
    name: str
    dist: np.ndarray
    time: np.ndarray
    demand: np.ndarray
    tw_start: np.ndarray
    tw_end: np.ndarray
    service: np.ndarray
    capacity: float
    num_vehicles: int
    horizon: float
    coords: np.ndarray
    size: int   # số khách hàng (không kể depot)


class _FuelLike:
    name = "khoảng cách"


class DistanceModel:
    """Bộ tiêu chí 'nhiên liệu = quãng đường' để dùng lại ACOSolver/metrics cho
    benchmark Solomon (mục tiêu kinh điển là tổng quãng đường)."""
    fuel = _FuelLike()

    def arc_liters(self, dist_m: float, load: float, speed_mps: float) -> float:
        return dist_m

    def optimal_speed(self) -> float:
        return 1.0

    def to_cost(self, liters: float) -> float:
        return liters

    def to_co2(self, liters: float) -> float:
        return 0.0


def load_solomon(path: str, size: Optional[int] = None) -> BenchmarkInstance:
    """Đọc file Solomon chuẩn. `size` = số khách lấy (25/50/100; None = tất cả).
    Các tập con 25/50 khách đầu cũng là benchmark chính thức của Solomon."""
    with open(path) as f:
        lines = [ln.rstrip() for ln in f]

    name = lines[0].strip()
    cap = num_veh = None
    rows: List[List[float]] = []
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("NUMBER"):
            parts = lines[idx + 1].split()
            num_veh, cap = int(parts[0]), float(parts[1])
        p = ln.split()
        if len(p) >= 7:
            try:
                int(p[0])                      # dòng dữ liệu: token đầu là số TT khách
            except ValueError:
                continue                       # bỏ qua các dòng tiêu đề
            rows.append([float(x) for x in p[:7]])

    # rows: [cust_no, x, y, demand, ready, due, service]; rows[0] = depot.
    if size is not None:
        rows = rows[:size + 1]
    n = len(rows)
    coords = np.array([[r[1], r[2]] for r in rows])
    demand = np.array([r[3] for r in rows])
    tw_start = np.array([r[4] for r in rows])
    tw_end = np.array([r[5] for r in rows])
    service = np.array([r[6] for r in rows])

    dist = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a != b:
                dist[a][b] = math.hypot(coords[a][0] - coords[b][0],
                                        coords[a][1] - coords[b][1])
    time = dist.copy()  # quy ước Solomon: thời gian = khoảng cách Euclid

    return BenchmarkInstance(
        name=name, dist=dist, time=time, demand=demand, tw_start=tw_start,
        tw_end=tw_end, service=service, capacity=cap, num_vehicles=num_veh,
        horizon=float(tw_end[0]), coords=coords, size=n - 1)


def _vehicles(inst: BenchmarkInstance) -> List[VehicleState]:
    return [VehicleState(vid=k, start_index=0, start_time=0.0, capacity=inst.capacity)
            for k in range(inst.num_vehicles)]


def _distance_of(routes, dist) -> float:
    total = 0.0
    for r in routes:
        for a, b in zip(r[:-1], r[1:]):
            total += dist[a][b]
    return total


def run_on_instance(inst: BenchmarkInstance, algo: str,
                    aco_cfg: Optional[ACOConfig] = None, seed: int = 1) -> Dict:
    """Chạy một thuật toán trên instance Solomon. algo ∈ {aco, nn, nn2opt, ortools}."""
    model = DistanceModel()
    cfg = aco_cfg or ACOConfig()
    pending = list(range(1, inst.size + 1))
    vehs = _vehicles(inst)
    solver = ACOSolver(cfg, model, congestion=None, depot_index=0,
                       rng=np.random.default_rng(seed))
    solver.bind(inst.time, inst.dist, inst.demand, inst.tw_start, inst.tw_end,
                inst.service, inst.horizon)

    t0 = _time.perf_counter()
    if algo == "aco":
        sol, _, _ = solver.solve(pending, vehs, inst.time, inst.dist, inst.demand,
                                 inst.tw_start, inst.tw_end, inst.service, inst.horizon)
    elif algo == "nn":
        sol = baseline.nearest_neighbor(pending, vehs, solver)
    elif algo == "nn2opt":
        sol = baseline.nearest_neighbor_2opt(pending, vehs, solver)
    elif algo == "ortools":
        sol = baseline.or_tools_solve(pending, vehs, solver)
        if sol is None:
            return {"algo": algo, "available": False}
    else:
        raise ValueError(algo)
    dt = _time.perf_counter() - t0

    used = [r for r in sol.routes if len(r) > 2]
    return {
        "algo": algo, "available": True,
        "vehicles": len(used),
        "distance": round(_distance_of(sol.routes, inst.dist), 2),
        "served": len(sol.served), "unserved": len(sol.unserved),
        "feasible": sol.feasible, "time_s": round(dt, 2),
    }
