"""Thí nghiệm 2 — ĐỊNH TUYẾN ĐỘNG: rolling-horizon với đơn phát sinh, tắc
đường và thời tiết, tái tối ưu bằng ACO.

So sánh hai chế độ để làm nổi bật giá trị của tái tối ưu:
  (A) ĐỘNG  : tái tối ưu mỗi khi có sự kiện/đến chu kỳ.
  (B) TĨNH  : chỉ lập kế hoạch 1 lần đầu ca (replan_interval rất lớn).

Chạy:  python -m experiments.run_dynamic
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CFG
from src import problem, visualize
from src.dynamic_sim import run_dynamic

OUT = "outputs"


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== Thí nghiệm định tuyến ĐỘNG (rolling-horizon + ACO) ===")
    inst = problem.generate_instance(
        CFG.map, CFG.problem,
        num_dynamic=CFG.dynamic.num_dynamic_orders,
        dynamic_seed=CFG.dynamic.seed)
    n_dyn = sum(1 for o in inst.orders if o.release_time > 0)
    print(f"Bản đồ thật (OSM): {inst.used_real_map}")
    print(f"{inst.num_nodes-1} đơn ({inst.num_nodes-1-n_dyn} cố định + "
          f"{n_dyn} phát sinh), {len(inst.vehicles)} xe\n")

    # (A) Chế độ ĐỘNG
    inst_a = copy.deepcopy(inst)
    res_a = run_dynamic(inst_a, CFG)
    print("--- (A) ĐỘNG: tái tối ưu liên tục ---")
    print(res_a.summary())

    # (B) Chế độ TĨNH: lập kế hoạch 1 lần (đặt chu kỳ tái tối ưu = vô cực).
    cfg_b = copy.deepcopy(CFG)
    cfg_b.dynamic.replan_interval_s = CFG.problem.horizon_s * 10  # chỉ 1 epoch
    inst_b = copy.deepcopy(inst)
    res_b = run_dynamic(inst_b, cfg_b)
    print("\n--- (B) TĨNH: chỉ lập kế hoạch đầu ca ---")
    print(res_b.summary())

    # So sánh — dùng nhiên liệu/đơn để công bằng (tĩnh giao ít đơn hơn nên
    # tổng nhiên liệu thấp hơn là điều hiển nhiên, không phải ưu điểm).
    print("\n=== SO SÁNH (ĐỘNG vs TĨNH) ===")
    d_served = res_a.served - res_b.served
    print(f"Đơn giao thêm được nhờ tái tối ưu : {d_served:+d} "
          f"(động {res_a.served} vs tĩnh {res_b.served})")
    fpo_a = res_a.fuel_liters / max(1, res_a.served)
    fpo_b = res_b.fuel_liters / max(1, res_b.served)
    print(f"Nhiên liệu / đơn (động)           : {fpo_a*1000:.1f} ml/đơn")
    print(f"Nhiên liệu / đơn (tĩnh)           : {fpo_b*1000:.1f} ml/đơn")
    if fpo_b > 0:
        ratio = 100.0 * (fpo_b - fpo_a) / fpo_b
        print(f"Hiệu quả nhiên liệu/đơn            : {ratio:+.1f}% "
              f"(dương = động tiết kiệm hơn trên mỗi đơn)")

    p = visualize.plot_executed(inst_a, res_a.executed_legs,
                                f"{OUT}/dynamic_executed.png")
    print(f"\nĐã lưu: {p}")


if __name__ == "__main__":
    main()
