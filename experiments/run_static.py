"""Thí nghiệm 1 — HYBRID ACO TĨNH (PRP + Time-Dependent VRPTW).

Giải với toàn bộ đơn biết trước; mục tiêu = nhiên liệu theo mô hình phát thải
phụ thuộc tốc độ, thời gian di chuyển theo giờ cao điểm. So sánh với baseline
(nearest-neighbor, NN+2opt, và OR-Tools nếu có).

Chạy:  python -m experiments.run_static
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from config import CFG
from src import problem, metrics, visualize, baseline
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion
from src.dynamic_sim import _order_arrays

OUT = "outputs"


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Hybrid ACO tĩnh (PRP + Time-Dependent VRPTW, mục tiêu nhiên liệu) ===")
    inst = problem.generate_instance(CFG.map, CFG.problem)
    print(f"Bản đồ thật (OSM): {inst.used_real_map} | "
          f"{inst.num_nodes-1} khách hàng, {len(inst.vehicles)} xe")

    em = EmissionModel(CFG.prp, CFG.fuel)
    cong = Congestion(CFG.traffic)
    print(f"Tốc độ tối ưu lý thuyết: {em.optimal_speed()*3.6:.1f} km/h | "
          f"Nhiên liệu: {em.fuel.name}\n")

    demand, tw_s, tw_e, serv = _order_arrays(inst)
    pending = [o.cid for o in inst.orders]
    states = [VehicleState(vid=k, start_index=0, start_time=0.0,
                           capacity=CFG.problem.vehicle_capacity)
              for k in range(CFG.problem.num_vehicles)]
    start_times = [0.0] * CFG.problem.num_vehicles

    solver = ACOSolver(CFG.aco, em, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(1))

    def run(sol):
        return metrics.evaluate(sol, inst.dist_matrix, inst.time_matrix, demand,
                                tw_s, tw_e, serv, em, congestion=cong,
                                horizon=CFG.problem.horizon_s,
                                vehicle_start_times=start_times)

    # --- Hybrid ACO ---
    sol, history, _ = solver.solve(pending, states, inst.time_matrix,
                                   inst.dist_matrix, demand, tw_s, tw_e, serv,
                                   CFG.problem.horizon_s)
    m_aco = run(sol)
    print("--- Hybrid ACO ---")
    print(m_aco.summary())
    print(f"Khả thi (phục vụ hết): {sol.feasible}\n")

    # --- Baselines ---
    m_nn = run(baseline.nearest_neighbor(pending, states, solver))
    m_nn2 = run(baseline.nearest_neighbor_2opt(pending, states, solver))
    ort = baseline.or_tools_solve(pending, states, solver)

    print("=== SO SÁNH (nhiên liệu, lít — thấp hơn = tốt hơn) ===")
    rows = [("Nearest-Neighbor", m_nn), ("NN + 2-opt", m_nn2),
            ("Hybrid ACO", m_aco)]
    if ort is not None:
        rows.insert(2, ("OR-Tools (GLS, 5s)", run(ort)))
    base = m_nn.fuel_liters
    for name, mm in rows:
        gap = 100.0 * (mm.fuel_liters - base) / base
        print(f"{name:<22}{mm.fuel_liters:>7.3f} lít  "
              f"({gap:+5.1f}% vs NN)  CO2={mm.co2_kg:.2f}kg  xe={mm.num_vehicles_used}")
    if ort is None:
        print("(OR-Tools chưa cài — bỏ qua. `pip install ortools` để bật baseline này.)")

    p1 = visualize.plot_routes(inst, sol.routes, f"{OUT}/static_routes.png",
                               "Hybrid ACO — lộ trình tối ưu nhiên liệu (PRP)")
    p2 = visualize.plot_convergence(history, f"{OUT}/static_convergence.png")
    print(f"\nĐã lưu: {p1}\n        {p2}")


if __name__ == "__main__":
    main()
