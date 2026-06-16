"""Tính các chỉ số đánh giá cho một lời giải định tuyến (PRP + tắc đường)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np

from src.aco_solver import Solution
from src.emission_model import EmissionModel


@dataclass
class RouteMetrics:
    num_vehicles_used: int
    total_distance_km: float
    total_time_h: float
    fuel_liters: float
    fuel_cost: float
    co2_kg: float
    fuel_name: str
    avg_speed_kph: float
    served: int
    unserved: int
    tw_violations: int

    def as_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Xe sử dụng      : {self.num_vehicles_used}\n"
            f"Tổng quãng đường: {self.total_distance_km:.2f} km\n"
            f"Tổng thời gian  : {self.total_time_h:.2f} giờ "
            f"(tốc độ TB {self.avg_speed_kph:.1f} km/h)\n"
            f"Nhiên liệu      : {self.fuel_liters:.2f} lít {self.fuel_name} "
            f"(~{self.fuel_cost:,.0f} đ)\n"
            f"Phát thải CO2   : {self.co2_kg:.2f} kg\n"
            f"Đã giao / Sót   : {self.served} / {self.unserved}\n"
            f"Vi phạm cửa sổ TG: {self.tw_violations}"
        )


def evaluate(sol: Solution, dist_m: np.ndarray, time_m: np.ndarray,
             demands: np.ndarray, tw_start: np.ndarray, tw_end: np.ndarray,
             service: np.ndarray, emission: EmissionModel,
             congestion: Optional[Callable[[float], float]] = None,
             horizon: float = 1e12,
             vehicle_start_times: Optional[List[float]] = None) -> RouteMetrics:
    cong = congestion or (lambda _t: 1.0)
    total_dist = total_time = total_drive = total_liters = 0.0
    used = 0
    tw_viol = 0
    served = 0

    for vi, route in enumerate(sol.routes):
        custs = route[1:-1]
        if not custs:
            continue
        used += 1
        served += len(custs)
        load = sum(demands[c] for c in custs)
        t0 = vehicle_start_times[vi] if vehicle_start_times else 0.0
        t = t0
        for a, b in zip(route[:-1], route[1:]):
            d = dist_m[a][b]
            eff = time_m[a][b] * cong(t)
            v = d / eff if eff > 1e-6 else emission.optimal_speed()
            total_dist += d
            total_drive += eff               # thời gian LÁI thực (không gồm chờ/phục vụ)
            total_liters += emission.arc_liters(d, load, v)
            arrival = t + eff
            if b != route[-1]:
                if arrival > tw_end[b] + 1e-6:
                    tw_viol += 1
                t = max(arrival, tw_start[b]) + service[b]
                load -= demands[b]
            else:
                t = arrival
        total_time += t - t0                 # thời gian tổng (gồm chờ cửa sổ + phục vụ)

    avg_speed = (total_dist / total_drive * 3.6) if total_drive > 0 else 0.0
    return RouteMetrics(
        num_vehicles_used=used,
        total_distance_km=total_dist / 1000.0,
        total_time_h=total_time / 3600.0,
        fuel_liters=total_liters,
        fuel_cost=emission.to_cost(total_liters),
        co2_kg=emission.to_co2(total_liters),
        fuel_name=emission.fuel.name,
        avg_speed_kph=avg_speed,
        served=served,
        unserved=len(sol.unserved),
        tw_violations=tw_viol,
    )
