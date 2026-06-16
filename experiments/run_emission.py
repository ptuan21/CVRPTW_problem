"""Thí nghiệm 4 — PHÂN TÍCH PHÁT THẢI THEO TỐC ĐỘ & GIỜ CAO ĐIỂM.

(a) Vẽ đường cong tốc độ–phát thải (PRP) và tốc độ tối ưu v*.
(b) Định lượng "tắc đường -> tốn nhiên liệu": giải cùng bài toán ở chế độ giao
    thông THÔNG THOÁNG vs GIỜ CAO ĐIỂM, so sánh nhiên liệu & CO2.

Chạy:  python -m experiments.run_emission
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from config import CFG
from src import problem, metrics, visualize
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion, flat_congestion
from src.dynamic_sim import _order_arrays

OUT = "outputs"


def main():
    os.makedirs(OUT, exist_ok=True)
    em = EmissionModel(CFG.prp, CFG.fuel)
    print("=== (a) Đường cong tốc độ – phát thải (PRP) ===")
    print(f"Tốc độ tối ưu v* = {em.optimal_speed()*3.6:.1f} km/h")
    p = visualize.plot_speed_emission(em, f"{OUT}/speed_emission_curve.png")
    print(f"Đã lưu: {p}\n")

    print("=== (b) Tác động của giờ cao điểm lên CÙNG MỘT bộ lộ trình ===")
    inst = problem.generate_instance(CFG.map, CFG.problem)
    em = EmissionModel(CFG.prp, CFG.fuel)
    cong = Congestion(CFG.traffic)
    demand, tw_s, tw_e, serv = _order_arrays(inst)
    pending = [o.cid for o in inst.orders]
    states = [VehicleState(vid=k, start_index=0, start_time=0.0,
                           capacity=CFG.problem.vehicle_capacity)
              for k in range(CFG.problem.num_vehicles)]
    # Tối ưu lộ trình một lần (có tính tới tắc đường).
    solver = ACOSolver(CFG.aco, em, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(1))
    sol, _, _ = solver.solve(pending, states, inst.time_matrix, inst.dist_matrix,
                             demand, tw_s, tw_e, serv, CFG.problem.horizon_s)

    # Cùng lộ trình, đánh giá dưới 2 chế độ giao thông để cô lập tác động.
    def ev(congestion):
        return metrics.evaluate(sol, inst.dist_matrix, inst.time_matrix, demand,
                                tw_s, tw_e, serv, em, congestion=congestion,
                                horizon=1e12,
                                vehicle_start_times=[0.0] * CFG.problem.num_vehicles)

    m_free = ev(flat_congestion)
    m_peak = ev(cong)
    print(f"{'Chế độ':<18}{'km/h':>7}{'Lít':>9}{'CO2(kg)':>10}")
    print(f"{'Thông thoáng':<18}{m_free.avg_speed_kph:>7.1f}"
          f"{m_free.fuel_liters:>9.3f}{m_free.co2_kg:>10.2f}")
    print(f"{'Có giờ cao điểm':<18}{m_peak.avg_speed_kph:>7.1f}"
          f"{m_peak.fuel_liters:>9.3f}{m_peak.co2_kg:>10.2f}")
    dl = 100.0 * (m_peak.fuel_liters - m_free.fuel_liters) / m_free.fuel_liters
    print(f"\n=> Tắc đường giờ cao điểm làm nhiên liệu thay đổi {dl:+.1f}% trên cùng "
          f"lộ trình\n   (tốc độ tụt khỏi vùng tối ưu {em.optimal_speed()*3.6:.0f} km/h "
          f"-> tốn xăng hơn).")


if __name__ == "__main__":
    main()
