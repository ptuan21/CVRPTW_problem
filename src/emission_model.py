"""Mô hình phát thải vi mô theo TỐC ĐỘ — Pollution-Routing Problem (PRP).

Thay cho mô hình tuyến tính chỉ theo tải trọng, mô-đun này dùng hàm tiêu thụ
nhiên liệu toàn diện (Bektaş & Laporte 2011; Demir et al. 2012; CMEM của Barth
et al.). Nhiên liệu trên một cung phụ thuộc TỐC ĐỘ + tải + gia tốc + độ dốc,
nên tồn tại một TỐC ĐỘ TỐI ƯU hình chữ U:
  - đi quá chậm -> tốn nhiên liệu nền của động cơ (số hạng k·Nₑ·Vₑ·d/v),
  - đi quá nhanh -> tốn lực cản khí động (số hạng β·d·v²).

Đây là chiều sâu khoa học cho phần nhiên liệu/CO2: giờ cao điểm làm tốc độ giảm
-> nhiên liệu tăng (khép kín "tắc đường -> tốn xăng"), và loại nhiên liệu E10
tác động qua hệ số nhiệt trị + phát thải.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from config import PRPConfig, FuelConfig, FuelType, FUEL_TYPES


class EmissionModel:
    def __init__(self, prp: PRPConfig, fuel_cfg: FuelConfig,
                 fuel_type: Optional[FuelType] = None):
        self.p = prp
        self.cfg = fuel_cfg
        self.fuel: FuelType = fuel_type or FUEL_TYPES[fuel_cfg.fuel_type]

        # Gộp các hằng số theo công thức PRP.
        self.lam = prp.xi / (prp.kappa * prp.psi)          # λ = ξ/(κ·ψ)
        self.gamma = 1.0 / (1000.0 * prp.eta_tf * prp.eta_eng)
        self.beta = 0.5 * prp.Cd * prp.rho_air * prp.area  # β = ½·C_d·ρ·A
        self.alpha = (prp.accel + prp.g * math.sin(prp.road_angle)
                      + prp.g * prp.Cr * math.cos(prp.road_angle))
        self.engine_kj = prp.k_eng * prp.Ne * prp.Ve       # k·Nₑ·Vₑ

    # ------------------------------------------------------------------
    # Lượng nhiên liệu (lít) cho MỘT cung
    # ------------------------------------------------------------------
    def arc_liters(self, dist_m: float, load: float, speed_mps: float) -> float:
        """Lít nhiên liệu thực tế trên cung dài dist_m, tải `load`, tốc độ
        speed_mps. Đã nhân hệ số nhiệt trị của loại nhiên liệu (E10...)."""
        v = max(speed_mps, 1e-3)
        total_mass = self.cfg.curb_weight + load
        engine = self.lam * self.engine_kj * (dist_m / v)
        traction = self.lam * self.gamma * self.alpha * total_mass * dist_m
        aero = self.lam * self.gamma * self.beta * dist_m * (v ** 2)
        liters_e0 = engine + traction + aero
        return liters_e0 * self.fuel.energy_factor

    def fuel_per_m(self, load: float, speed_mps: float) -> float:
        """Lít/mét ở tốc độ cho trước (tiện vẽ đường cong tốc độ-phát thải)."""
        return self.arc_liters(1.0, load, speed_mps)

    # ------------------------------------------------------------------
    # Tốc độ tối ưu (cực tiểu nhiên liệu/quãng đường) — nghiệm giải tích
    # ------------------------------------------------------------------
    def optimal_speed(self) -> float:
        """v* = (k·Nₑ·Vₑ / (2·γ·β))^(1/3), kẹp trong [v_min, v_max].

        Suy ra từ d/dv[ engine/v + aero·v² ] = 0 (số hạng tải không phụ thuộc v)."""
        v_star = (self.engine_kj / (2.0 * self.gamma * self.beta)) ** (1.0 / 3.0)
        return float(min(max(v_star, self.p.v_min), self.p.v_max))

    # ------------------------------------------------------------------
    # Lượng nhiên liệu cho một tuyến / lời giải
    # ------------------------------------------------------------------
    def route_liters(self, route: List[int], dist_matrix: np.ndarray,
                     time_matrix: np.ndarray, demands: np.ndarray) -> float:
        """Tổng nhiên liệu cho một tuyến. Tốc độ mỗi cung suy ra từ
        quãng đường / thời gian (đã bao gồm tắc đường nếu time_matrix động)."""
        if len(route) < 2:
            return 0.0
        load = sum(demands[c] for c in route)
        total = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d = dist_matrix[a][b]
            t = time_matrix[a][b]
            v = d / t if t > 1e-6 else self.optimal_speed()
            total += self.arc_liters(d, load, v)
            load -= demands[b]
        return total

    def solution_liters(self, routes: List[List[int]], dist_matrix: np.ndarray,
                        time_matrix: np.ndarray, demands: np.ndarray) -> float:
        return sum(self.route_liters(r, dist_matrix, time_matrix, demands)
                   for r in routes)

    # ------------------------------------------------------------------
    # Quy đổi chi phí & CO2
    # ------------------------------------------------------------------
    def to_cost(self, liters: float) -> float:
        return liters * self.fuel.price

    def to_co2(self, liters: float) -> float:
        return liters * self.fuel.co2_kg_per_liter
