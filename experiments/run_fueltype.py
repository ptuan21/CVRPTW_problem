"""Thí nghiệm 3 — SO SÁNH LOẠI NHIÊN LIỆU (E0 / E5 / E10).

Tối ưu lộ trình MỘT LẦN (theo E10 — chuẩn Hà Nội 2026), rồi đánh giá CÙNG bộ
lộ trình đó dưới ba loại nhiên liệu (mô hình phát thải PRP) để cô lập đúng tác
động của nhiên liệu lên thể tích tiêu thụ, chi phí và CO2 hoá thạch.

Chạy:  python -m experiments.run_fueltype
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from config import CFG, FUEL_TYPES
from src import problem
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion
from src.dynamic_sim import _order_arrays


def main():
    print("=== So sánh loại nhiên liệu trên cùng lộ trình tối ưu (mô hình PRP) ===\n")
    inst = problem.generate_instance(CFG.map, CFG.problem)
    demand, tw_s, tw_e, serv = _order_arrays(inst)
    cong = Congestion(CFG.traffic)
    pending = [o.cid for o in inst.orders]
    states = [VehicleState(vid=k, start_index=0, start_time=0.0,
                           capacity=CFG.problem.vehicle_capacity)
              for k in range(CFG.problem.num_vehicles)]

    # Tối ưu theo E10 (mặc định cấu hình).
    em10 = EmissionModel(CFG.prp, CFG.fuel, FUEL_TYPES["E10"])
    solver = ACOSolver(CFG.aco, em10, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(1))
    sol, _, _ = solver.solve(pending, states, inst.time_matrix, inst.dist_matrix,
                             demand, tw_s, tw_e, serv, CFG.problem.horizon_s)
    print(f"Lộ trình tối ưu theo E10: phục vụ {len(sol.served)} đơn, "
          f"khả thi={sol.feasible}\n")

    header = f"{'Loại':<26}{'Lít':>9}{'Chi phí (đ)':>15}{'CO2 (kg)':>12}"
    print(header)
    print("-" * len(header))
    base = None
    for key in ("E0", "E5", "E10"):
        em = EmissionModel(CFG.prp, CFG.fuel, FUEL_TYPES[key])
        liters = em.solution_liters(sol.routes, inst.dist_matrix,
                                    inst.time_matrix, demand)
        cost, co2 = em.to_cost(liters), em.to_co2(liters)
        if key == "E0":
            base = (liters, cost, co2)
        print(f"{FUEL_TYPES[key].name:<26}{liters:>9.3f}{cost:>15,.0f}{co2:>12.3f}")

    em10b = EmissionModel(CFG.prp, CFG.fuel, FUEL_TYPES["E10"])
    l10 = em10b.solution_liters(sol.routes, inst.dist_matrix, inst.time_matrix, demand)
    c10, q10 = em10b.to_cost(l10), em10b.to_co2(l10)
    bl, bc, bq = base
    print("\n--- E10 so với xăng khoáng E0 ---")
    print(f"Thể tích : {100*(l10-bl)/bl:+.1f}%  (E10 tốn thêm vì nhiệt trị thấp hơn)")
    print(f"Chi phí  : {100*(c10-bc)/bc:+.1f}%  (giá E10 rẻ hơn nên có thể bù lại)")
    print(f"CO2      : {100*(q10-bq)/bq:+.1f}%  (E10 giảm phát thải hoá thạch)")


if __name__ == "__main__":
    main()
