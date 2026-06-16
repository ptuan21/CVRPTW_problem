"""Mô phỏng định tuyến ĐỘNG theo cơ chế rolling-horizon.

Vòng lặp theo từng epoch (chu kỳ tái tối ưu):
  1. Thực thi kế hoạch hiện tại tới thời điểm `now` (xe chạy, giao hàng).
  2. Hé lộ các đơn mới phát sinh (release_time <= now).
  3. Áp sự kiện ngẫu nhiên: tắc đường theo vùng + thời tiết toàn mạng
     -> nhân hệ số vào ma trận THỜI GIAN (không đổi quãng đường/nhiên liệu).
  4. Tái tối ưu bằng ACO cho toàn bộ đơn đã biết & chưa giao, với mỗi xe
     xuất phát từ vị trí & thời điểm hiện tại của nó.

Giả định đơn giản hoá: giữa hai epoch, xe chỉ "chốt"
các điểm đã giao xong trước `now`; nếu đang trên đường tới điểm kế, điểm đó sẽ
được tái lập kế hoạch ở epoch sau. Tải nhiên liệu của một chặng tính theo lượng
hàng còn trên xe của kế hoạch đang chạy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import time as _time

import numpy as np

from config import Config
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion
from src.problem import ProblemInstance, known_orders


@dataclass
class ExecutedLeg:
    a: int
    b: int
    load: float
    dist: float
    vid: int = 0


@dataclass
class DynamicResult:
    served: int
    unserved: int
    fuel_liters: float
    fuel_cost: float
    co2_kg: float
    fuel_name: str
    total_distance_km: float
    num_replans: int
    total_solve_time_s: float
    avg_solve_time_s: float
    executed_legs: List[ExecutedLeg] = field(default_factory=list)
    vehicle_plans: List[List[int]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Đơn đã giao     : {self.served}\n"
            f"Đơn không kịp   : {self.unserved}\n"
            f"Tổng quãng đường: {self.total_distance_km:.2f} km\n"
            f"Nhiên liệu      : {self.fuel_liters:.2f} lít {self.fuel_name} "
            f"(~{self.fuel_cost:,.0f} đ)\n"
            f"Phát thải CO2   : {self.co2_kg:.2f} kg\n"
            f"Số lần tái tối ưu: {self.num_replans}\n"
            f"Thời gian giải TB: {self.avg_solve_time_s*1000:.0f} ms/lần"
        )


def _order_arrays(inst: ProblemInstance):
    n = inst.num_nodes
    demand = np.zeros(n)
    tw_s = np.zeros(n)
    tw_e = np.full(n, inst.cfg.horizon_s)
    serv = np.zeros(n)
    for o in inst.orders:
        demand[o.cid] = o.demand
        tw_s[o.cid] = o.tw_start
        tw_e[o.cid] = o.tw_end
        serv[o.cid] = o.service_time
    return demand, tw_s, tw_e, serv


def _apply_traffic(time_m: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Trả về ma trận thời gian đã nhân hệ số tắc đường + thời tiết cho epoch này."""
    scaled = time_m * cfg.dynamic.weather_multiplier
    if rng.random() < cfg.dynamic.traffic_event_prob:
        n = scaled.shape[0]
        # Chọn một "vùng tắc": ~30% số node; mọi cung đi vào/ra vùng bị chậm.
        k = max(1, int(0.3 * n))
        zone = rng.choice(n, size=k, replace=False)
        m = cfg.dynamic.traffic_multiplier
        scaled[np.ix_(zone, range(n))] *= m
        scaled[np.ix_(range(n), zone)] *= m
    return scaled


def run_dynamic(inst: ProblemInstance, cfg: Config) -> DynamicResult:
    em = EmissionModel(cfg.prp, cfg.fuel)
    cong = Congestion(cfg.traffic)
    demand, tw_s, tw_e, serv = _order_arrays(inst)
    horizon = cfg.problem.horizon_s
    rng = np.random.default_rng(cfg.dynamic.seed)
    solver = ACOSolver(cfg.aco, em, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(cfg.dynamic.seed + 1))

    # Trạng thái xe: vị trí, thời điểm rảnh, kế hoạch (danh sách cid chưa giao).
    veh_pos = [0] * cfg.problem.num_vehicles
    veh_time = [0.0] * cfg.problem.num_vehicles
    veh_plan: List[List[int]] = [[] for _ in range(cfg.problem.num_vehicles)]
    veh_onboard: List[float] = [0.0] * cfg.problem.num_vehicles

    executed: List[ExecutedLeg] = []
    served_ids: set = set()
    num_replans = 0
    solve_time = 0.0
    total_fuel = 0.0
    pheromone = None  # warm-start giữa các epoch

    def execute_until(now: float):
        """Chạy kế hoạch của mọi xe tới thời điểm `now`, chốt điểm đã giao xong.
        Thời gian & nhiên liệu mỗi chặng tính theo tắc đường tại GIỜ KHỞI HÀNH."""
        nonlocal total_fuel
        for k in range(cfg.problem.num_vehicles):
            pos, t = veh_pos[k], veh_time[k]
            plan = veh_plan[k]
            onboard = veh_onboard[k]
            while plan:
                nxt = plan[0]
                eff = inst.time_matrix[pos][nxt] * cong(t)
                arrival = t + eff
                completion = max(arrival, tw_s[nxt]) + serv[nxt]
                if completion <= now:
                    d = inst.dist_matrix[pos][nxt]
                    v = d / eff if eff > 1e-6 else em.optimal_speed()
                    total_fuel += em.arc_liters(d, onboard, v)
                    executed.append(ExecutedLeg(pos, nxt, onboard, d, vid=k))
                    onboard -= demand[nxt]
                    served_ids.add(nxt)
                    for o in inst.orders:
                        if o.cid == nxt:
                            o.served = True
                    pos, t = nxt, completion
                    plan.pop(0)
                else:
                    break
            veh_pos[k], veh_time[k], veh_onboard[k] = pos, t, onboard

    # ----- Vòng lặp các epoch -----
    epoch = 0.0
    while epoch <= horizon:
        execute_until(epoch)

        pending = [o.cid for o in known_orders(inst, epoch)]
        if pending:
            # Sự cố tắc ngẫu nhiên theo vùng (chồng lên hồ sơ giờ cao điểm trong solver).
            scaled_time = _apply_traffic(inst.time_matrix, cfg, rng)
            states = [VehicleState(vid=k, start_index=veh_pos[k],
                                   start_time=max(veh_time[k], epoch),
                                   capacity=cfg.problem.vehicle_capacity)
                      for k in range(cfg.problem.num_vehicles)]
            warm = pheromone if cfg.dynamic.warm_start_pheromone else None
            t0 = _time.perf_counter()
            sol, _, pheromone = solver.solve(pending, states, scaled_time,
                                             inst.dist_matrix, demand, tw_s, tw_e,
                                             serv, horizon, init_pheromone=warm)
            solve_time += _time.perf_counter() - t0
            num_replans += 1
            # Gán lại kế hoạch: phần giữa của mỗi route (bỏ start & depot cuối).
            for k, route in enumerate(sol.routes):
                veh_plan[k] = [c for c in route[1:-1]]
                veh_onboard[k] = sum(demand[c] for c in veh_plan[k])

        epoch += cfg.dynamic.replan_interval_s

    # Chạy nốt mọi kế hoạch còn lại tới hết (không giới hạn thời gian).
    execute_until(float("inf"))

    total_dist = sum(l.dist for l in executed)
    all_served = len(served_ids)
    total_orders = len(inst.orders)

    return DynamicResult(
        served=all_served,
        unserved=total_orders - all_served,
        fuel_liters=total_fuel,
        fuel_cost=em.to_cost(total_fuel),
        co2_kg=em.to_co2(total_fuel),
        fuel_name=em.fuel.name,
        total_distance_km=total_dist / 1000.0,
        num_replans=num_replans,
        total_solve_time_s=solve_time,
        avg_solve_time_s=solve_time / max(1, num_replans),
        executed_legs=executed,
        vehicle_plans=veh_plan,
    )
