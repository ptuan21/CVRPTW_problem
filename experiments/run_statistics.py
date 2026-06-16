"""Thí nghiệm 6 — ĐÁNH GIÁ THỐNG KÊ ĐA-SEED (độ tin cậy học thuật).

Sinh N instance Hà Nội độc lập (khác seed), chạy mỗi thuật toán trên CÙNG instance
(thiết kế bắt cặp), rồi báo cáo:
  - trung bình ± khoảng tin cậy 95% (phân phối t) cho nhiên liệu, quãng đường, số xe;
  - kiểm định Wilcoxon signed-rank (bắt cặp) so Hybrid ACO với từng baseline;
  - tỉ lệ thắng theo từng instance + mức cải thiện trung bình.

Chạy:  python -m experiments.run_statistics [N]      (mặc định N=10)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scipy.stats as st

from config import CFG
from src import problem, metrics, graph_loader, baseline
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion
from src.dynamic_sim import _order_arrays

OUT = "outputs"
ALGOS = [("Hybrid ACO", "aco"), ("NN+2-opt", "nn2opt"), ("OR-Tools", "ortools")]


def _solve(inst, cfg, algo):
    em = EmissionModel(cfg.prp, cfg.fuel)
    cong = Congestion(cfg.traffic)
    demand, tw_s, tw_e, serv = _order_arrays(inst)
    pending = [o.cid for o in inst.orders]
    states = [VehicleState(vid=k, start_index=0, start_time=0.0,
                           capacity=cfg.problem.vehicle_capacity)
              for k in range(cfg.problem.num_vehicles)]
    solver = ACOSolver(cfg.aco, em, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(1))
    solver.bind(inst.time_matrix, inst.dist_matrix, demand, tw_s, tw_e, serv,
                cfg.problem.horizon_s)
    if algo == "aco":
        sol, _, _ = solver.solve(pending, states, inst.time_matrix, inst.dist_matrix,
                                 demand, tw_s, tw_e, serv, cfg.problem.horizon_s)
    elif algo == "nn2opt":
        sol = baseline.nearest_neighbor_2opt(pending, states, solver)
    elif algo == "ortools":
        sol = baseline.or_tools_solve(pending, states, solver)
        if sol is None:
            return None
    m = metrics.evaluate(sol, inst.dist_matrix, inst.time_matrix, demand, tw_s,
                         tw_e, serv, em, congestion=cong,
                         horizon=cfg.problem.horizon_s,
                         vehicle_start_times=[0.0] * cfg.problem.num_vehicles)
    return m


def ci95(data):
    a = np.asarray(data, dtype=float)
    m = a.mean()
    if len(a) < 2:
        return m, m, m
    se = st.sem(a)
    lo, hi = st.t.interval(0.95, len(a) - 1, loc=m, scale=se)
    return m, lo, hi


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    os.makedirs(OUT, exist_ok=True)
    print(f"=== Đánh giá thống kê đa-seed (N={n} instance Hà Nội, 25 đơn) ===\n")

    # Tải graph một lần, dùng lại cho mọi seed.
    G, used_real = graph_loader.load_or_build_graph(CFG.map)
    orig = graph_loader.load_or_build_graph
    graph_loader.load_or_build_graph = lambda _c: (G, used_real)

    cfg = copy.deepcopy(CFG)
    cfg.aco.num_ants = 20
    cfg.aco.num_iterations = 60

    fuel = {k: [] for _, k in ALGOS}
    dist = {k: [] for _, k in ALGOS}
    veh = {k: [] for _, k in ALGOS}
    ort_available = True

    try:
        for s in range(n):
            cfg.problem.seed = 1000 + s
            inst = problem.generate_instance(cfg.map, cfg.problem)
            for _, algo in ALGOS:
                m = _solve(inst, cfg, algo)
                if m is None:
                    ort_available = False
                    continue
                fuel[algo].append(m.fuel_liters)
                dist[algo].append(m.total_distance_km)
                veh[algo].append(m.num_vehicles_used)
            print(f"  seed {s+1}/{n} xong", end="\r")
    finally:
        graph_loader.load_or_build_graph = orig
    print(" " * 30, end="\r")

    # Bảng trung bình ± CI95.
    print(f"{'Thuật toán':<13}{'Nhiên liệu (L)':>22}{'Quãng đường (km)':>22}{'Số xe':>10}")
    print("-" * 67)
    for name, algo in ALGOS:
        if not fuel[algo]:
            print(f"{name:<13}{'(OR-Tools chưa cài)':>22}")
            continue
        fm, flo, fhi = ci95(fuel[algo])
        dm, dlo, dhi = ci95(dist[algo])
        vm, _, _ = ci95(veh[algo])
        print(f"{name:<13}{f'{fm:.3f} [{flo:.3f},{fhi:.3f}]':>22}"
              f"{f'{dm:.2f} [{dlo:.2f},{dhi:.2f}]':>22}{vm:>10.1f}")
    print("\n(giá trị: trung bình [khoảng tin cậy 95%])")

    # Kiểm định bắt cặp Wilcoxon: ACO vs từng baseline (trên nhiên liệu).
    print("\n=== Kiểm định Wilcoxon signed-rank — Hybrid ACO vs baseline (nhiên liệu) ===")
    aco = np.array(fuel["aco"])
    for name, algo in ALGOS:
        if algo == "aco" or not fuel[algo]:
            continue
        base = np.array(fuel[algo])
        diff = base - aco                      # >0 nghĩa là ACO tốt hơn (ít xăng hơn)
        wins = int((diff > 0).sum())
        improve = 100.0 * diff.mean() / base.mean()
        try:
            _, p = st.wilcoxon(aco, base)
            ptxt = f"p={p:.4g}"
            sig = "có ý nghĩa (p<0.05)" if p < 0.05 else "chưa có ý nghĩa"
        except ValueError:
            ptxt, sig = "p=N/A", "(các giá trị trùng nhau)"
        print(f"  vs {name:<10}: ACO thắng {wins}/{len(base)} instance, "
              f"cải thiện TB {improve:+.1f}%, {ptxt} -> {sig}")

    # Biểu đồ cột nhiên liệu trung bình ± CI95.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [nm for nm, a in ALGOS if fuel[a]]
        means = [np.mean(fuel[a]) for _, a in ALGOS if fuel[a]]
        errs = [np.mean(fuel[a]) - ci95(fuel[a])[1] for _, a in ALGOS if fuel[a]]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(names, means, yerr=errs, capsize=6,
               color=["#2c7", "#fa3", "#37c"][:len(names)])
        ax.set_ylabel("Nhiên liệu trung bình (lít)")
        ax.set_title(f"So sánh thuật toán trên {n} instance (± khoảng tin cậy 95%)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = f"{OUT}/statistics_fuel.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"\nĐã lưu biểu đồ: {path}")
    except Exception as exc:
        print(f"(Bỏ qua biểu đồ: {exc})")


if __name__ == "__main__":
    main()
