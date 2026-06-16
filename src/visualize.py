"""Trực quan hoá: vẽ lộ trình trên mặt phẳng toạ độ và đường hội tụ của ACO.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # backend không cần màn hình
import matplotlib.pyplot as plt
import numpy as np

from src.problem import ProblemInstance

_COLORS = plt.cm.tab10.colors


def plot_routes(inst: ProblemInstance, routes: Sequence[Sequence[int]],
                path: str, title: str = "Lộ trình giao hàng") -> str:
    fig, ax = plt.subplots(figsize=(9, 9))
    coords = inst.coords

    # Khách hàng & kho.
    ax.scatter(coords[1:, 0], coords[1:, 1], c="#444", s=30, zorder=3, label="Khách hàng")
    ax.scatter([coords[0, 0]], [coords[0, 1]], c="red", marker="*", s=300,
               zorder=4, label="Kho (depot)")

    used = 0
    for vi, route in enumerate(routes):
        pts = [c for c in route]
        if len(set(pts)) <= 1:
            continue
        color = _COLORS[used % len(_COLORS)]
        used += 1
        xs = coords[pts, 0]
        ys = coords[pts, 1]
        ax.plot(xs, ys, "-", color=color, alpha=0.8, lw=1.8,
                label=f"Xe {vi}", zorder=2)

    ax.set_title(f"{title}  ({used} xe)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_executed(inst: ProblemInstance, legs, path: str,
                  title: str = "Lộ trình đã thực thi (động)") -> str:
    """Vẽ các chặng đã thực sự chạy trong mô phỏng động."""
    fig, ax = plt.subplots(figsize=(9, 9))
    coords = inst.coords
    ax.scatter(coords[1:, 0], coords[1:, 1], c="#444", s=30, zorder=3, label="Khách hàng")
    ax.scatter([coords[0, 0]], [coords[0, 1]], c="red", marker="*", s=300,
               zorder=4, label="Kho")
    for leg in legs:
        ax.plot([coords[leg.a, 0], coords[leg.b, 0]],
                [coords[leg.a, 1], coords[leg.b, 1]],
                "-", color="#1f77b4", alpha=0.6, lw=1.5, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_speed_emission(em, path: str, loads=(0.0, 300.0, 600.0),
                        title: str = "Đường cong tốc độ – phát thải (PRP)") -> str:
    """Vẽ nhiên liệu (lít/100km) theo tốc độ ở vài mức tải; đánh dấu v* tối ưu."""
    fig, ax = plt.subplots(figsize=(8, 5))
    speeds_kph = np.linspace(8, 90, 60)
    for load in loads:
        y = [em.fuel_per_m(load, v / 3.6) * 100000 for v in speeds_kph]
        ax.plot(speeds_kph, y, label=f"tải {load:.0f} kg")
    v_star = em.optimal_speed() * 3.6
    ax.axvline(v_star, color="red", ls="--", lw=1)
    ax.text(v_star + 1, ax.get_ylim()[1] * 0.9, f"v* ≈ {v_star:.0f} km/h",
            color="red", fontsize=9)
    ax.set_xlabel("Tốc độ (km/h)")
    ax.set_ylabel("Tiêu thụ (lít/100km)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_convergence(history: List[float], path: str,
                     title: str = "Đường hội tụ ACO") -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history) + 1), history, "-o", ms=3, color="#2c7")
    ax.set_xlabel("Vòng lặp")
    ax.set_ylabel("Chi phí tốt nhất (nhiên liệu, lít + phạt)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
