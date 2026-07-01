import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ACOConfig
from src import benchmark as bm

INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"=== Benchmark Solomon VRPTW (mục tiêu: tổng quãng đường, size={size}) ===\n")
    cfg = ACOConfig()
    cfg.num_ants = 20
    cfg.num_iterations = 60

    algos = ["nn", "nn2opt", "aco", "ortools"]
    head = f"{'Instance':<9}" + "".join(f"{a:>16}" for a in
                                         ["NN", "NN+2opt", "Hybrid ACO", "OR-Tools"])
    if size == 100:
        head += f"{'BKS(tham khảo)':>16}"
    print(head)
    print("-" * len(head))

    aco_gaps = []
    chart = {a: [] for a in algos}       # quãng đường theo instance để vẽ
    chart_bks, names_done = [], []
    for name in INSTANCES:
        path = f"data/solomon/{name}.txt"
        if not os.path.exists(path):
            continue
        inst = bm.load_solomon(path, size=size)
        cells, dists = [], {}
        for a in algos:
            r = bm.run_on_instance(inst, a, cfg)
            if not r.get("available"):
                cells.append("—")
                chart[a].append(0)
                continue
            mark = "" if r["feasible"] else "*"   # * = vi phạm ràng buộc
            cells.append(f"{r['vehicles']}x/{r['distance']:.0f}{mark}")
            chart[a].append(r["distance"])
            if r["feasible"]:
                dists[a] = r["distance"]
        row = f"{name:<9}" + "".join(f"{c:>16}" for c in cells)
        bks_d = bm.BKS_100.get(name, (None, None))[1] if size == 100 else None
        if size == 100 and name in bm.BKS_100:
            v, d = bm.BKS_100[name]
            row += f"{f'{v}x/{d:.0f}':>16}"
            if "aco" in dists:
                aco_gaps.append(100 * (dists["aco"] - d) / d)
        chart_bks.append(bks_d or 0)
        names_done.append(name)
        print(row)

    print("\n(ô = sốXe / quãngĐường; * = vi phạm cửa sổ TG; thấp hơn = tốt hơn)")
    if aco_gaps:
        print(f"Khoảng cách ACO so với BKS (trung bình): "
              f"+{sum(aco_gaps)/len(aco_gaps):.1f}%")
    print("Ghi chú: solver dùng đội xe CỐ ĐỊNH + tối thiểu quãng đường, khác mục tiêu")
    print("phân cấp 'ít xe trước rồi mới tới quãng đường' của Solomon -> số xe có thể")
    print("lệch BKS. Bản 25/50 khách là tập con chính thức; BKS chỉ áp cho bản 100 khách.")

    # Vẽ biểu đồ cột nhóm quãng đường theo instance.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        os.makedirs("outputs", exist_ok=True)
        labels = {"nn": "NN", "nn2opt": "NN+2opt", "aco": "Hybrid ACO", "ortools": "OR-Tools"}
        series = [("nn2opt", "#f4a"), ("aco", "#2c7"), ("ortools", "#37c")]
        x = np.arange(len(names_done))
        w = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (a, col) in enumerate(series):
            ax.bar(x + (i - 1) * w, chart[a], w, label=labels[a], color=col)
        if size == 100 and any(chart_bks):
            ax.bar(x + 2 * w, chart_bks, w, label="BKS", color="#333", alpha=.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names_done)
        ax.set_ylabel("Tổng quãng đường")
        ax.set_title(f"Benchmark Solomon VRPTW (size={size}) — quãng đường (thấp hơn = tốt hơn)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = f"outputs/benchmark_solomon_{size}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Đã lưu biểu đồ: {path}")
    except Exception as exc:
        print(f"(Bỏ qua biểu đồ: {exc})")


if __name__ == "__main__":
    main()
