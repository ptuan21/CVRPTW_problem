"""Hybrid Ant Colony Optimization (MAX-MIN Ant System + Local Search) cho
bài toán định tuyến phát thải-động (PRP + Time-Dependent VRPTW).

Khác bản cơ bản:
  - Hàm mục tiêu = NHIÊN LIỆU theo mô hình PRP (phụ thuộc tốc độ), không phải
    quãng đường. Tốc độ mỗi cung suy ra từ thời gian di chuyển có tắc đường.
  - Thời gian di chuyển PHỤ THUỘC THỜI ĐIỂM: hệ số tắc theo giờ khởi hành.
  - Sau mỗi vòng, LOCAL SEARCH (2-opt + Or-opt + relocate) tinh chỉnh lời giải.
  - Hỗ trợ WARM-START: nhận & trả pheromone để tái dùng giữa các epoch động.

Solver tách rời instance để tái dùng trong mô phỏng động: mỗi xe xuất phát từ
một node-index và thời điểm bất kỳ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from config import ACOConfig
from src.emission_model import EmissionModel


@dataclass
class VehicleState:
    vid: int
    start_index: int
    start_time: float
    capacity: float


@dataclass
class Solution:
    routes: List[List[int]]
    served: List[int]
    unserved: List[int]
    liters: float
    feasible: bool
    cost: float = field(default=0.0)


class ACOSolver:
    def __init__(self, cfg: ACOConfig, emission: EmissionModel,
                 congestion: Optional[Callable[[float], float]] = None,
                 depot_index: int = 0, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.em = emission
        self.cong = congestion or (lambda _t: 1.0)
        self.depot = depot_index
        self.rng = rng or np.random.default_rng(0)
        # Các mảng bài toán, gán khi solve()/bind() để evaluator dùng chung.
        self._t = self._d = self._dem = None
        self._tw_s = self._tw_e = self._serv = None
        self._horizon = 0.0

    # ------------------------------------------------------------------
    def bind(self, time_matrix, dist_matrix, demands, tw_start, tw_end,
             service, horizon) -> None:
        """Gắn dữ liệu bài toán vào solver (để baseline tái dùng evaluator/feasible
        mà không cần chạy ACO)."""
        self._t, self._d, self._dem = time_matrix, dist_matrix, demands
        self._tw_s, self._tw_e, self._serv = tw_start, tw_end, service
        self._horizon = horizon

    def solve(self, pending: Sequence[int], vehicles: Sequence[VehicleState],
              time_matrix: np.ndarray, dist_matrix: np.ndarray,
              demands: np.ndarray, tw_start: np.ndarray, tw_end: np.ndarray,
              service: np.ndarray, horizon: float,
              init_pheromone: Optional[np.ndarray] = None
              ) -> Tuple[Solution, List[float], np.ndarray]:
        """Trả về (best_solution, history, pheromone) — pheromone để warm-start."""
        pending = list(pending)
        self.bind(time_matrix, dist_matrix, demands, tw_start, tw_end, service, horizon)
        n = time_matrix.shape[0]

        if not pending:
            empty = Solution([[v.start_index, self.depot] for v in vehicles],
                             [], [], 0.0, True, 0.0)
            return empty, [0.0], (init_pheromone if init_pheromone is not None
                                  else np.full((n, n), self.cfg.tau_max))

        tau = init_pheromone.copy() if init_pheromone is not None \
            else np.full((n, n), self.cfg.tau_max)
        eta = 1.0 / (dist_matrix + 1.0)

        best: Optional[Solution] = None
        history: List[float] = []

        for _ in range(self.cfg.num_iterations):
            iter_best: Optional[Solution] = None
            for _ant in range(self.cfg.num_ants):
                sol = self._construct(pending, vehicles, tau, eta)
                if iter_best is None or sol.cost < iter_best.cost:
                    iter_best = sol
            if self.cfg.local_search:
                # Mỗi vòng chỉ tinh chỉnh TRONG tuyến (rẻ) để dẫn dắt pheromone.
                iter_best = self._local_search(iter_best, vehicles, pending, "intra")
            if best is None or iter_best.cost < best.cost:
                best = iter_best
            self._update_pheromone(tau, best)
            history.append(best.cost)

        # Polish cuối: thêm các phép LIÊN TUYẾN (relocate/swap/2-opt*) để gom xe.
        if self.cfg.local_search and best is not None:
            best = self._local_search(best, vehicles, pending, "full")
            history.append(best.cost)

        return best, history, tau

    # ------------------------------------------------------------------
    # Dựng lời giải (greedy ngẫu nhiên, kiểm tra khả thi theo TD)
    # ------------------------------------------------------------------
    def _construct(self, pending, vehicles, tau, eta) -> Solution:
        unvisited = set(pending)
        routes: List[List[int]] = []
        for v in vehicles:
            route = [v.start_index]
            cur, t, load_used = v.start_index, v.start_time, 0.0
            while unvisited:
                cand = self._feasible(cur, t, load_used, v.capacity, unvisited)
                if not cand:
                    break
                nxt = self._choose(cur, cand, tau, eta)
                eff = self._t[cur][nxt] * self.cong(t)
                arrival = t + eff
                start_service = max(arrival, self._tw_s[nxt]) \
                    if self.cfg.early_wait_allowed else arrival
                t = start_service + self._serv[nxt]
                load_used += self._dem[nxt]
                cur = nxt
                route.append(nxt)
                unvisited.discard(nxt)
            route.append(self.depot)
            routes.append(route)
        return self._make_solution(routes, vehicles, pending)

    def _feasible(self, cur, t, load_used, capacity, unvisited):
        out = []
        cf = self.cong(t)
        for j in unvisited:
            if load_used + self._dem[j] > capacity:
                continue
            arrival = t + self._t[cur][j] * cf
            if arrival > self._tw_e[j] or arrival > self._horizon:
                continue
            out.append(j)
        return out

    def _choose(self, cur, cand, tau, eta) -> int:
        w = np.array([(tau[cur][j] ** self.cfg.alpha) * (eta[cur][j] ** self.cfg.beta)
                      for j in cand])
        if self.rng.random() < self.cfg.q0:
            return cand[int(np.argmax(w))]
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return cand[int(self.rng.integers(len(cand)))]
        return cand[int(self.rng.choice(len(cand), p=w / s))]

    # ------------------------------------------------------------------
    # Đánh giá một tuyến dưới PRP + tắc đường (nguồn chân lý duy nhất)
    # ------------------------------------------------------------------
    def _route_eval(self, route: List[int], start_time: float,
                    capacity: float) -> Tuple[float, bool]:
        """Trả về (liters, feasible). Tải mỗi cung = tổng demand các điểm còn
        phải giao; tốc độ = quãng đường / thời gian có tắc."""
        custs = route[1:-1]
        load = sum(self._dem[c] for c in custs)
        feasible = load <= capacity + 1e-9
        t, liters = start_time, 0.0
        for a, b in zip(route[:-1], route[1:]):
            eff = self._t[a][b] * self.cong(t)
            v = self._d[a][b] / eff if eff > 1e-6 else self.em.optimal_speed()
            liters += self.em.arc_liters(self._d[a][b], load, v)
            arrival = t + eff
            if b != route[-1]:
                if arrival > self._tw_e[b] + 1e-6 or arrival > self._horizon:
                    feasible = False
                start_service = max(arrival, self._tw_s[b]) \
                    if self.cfg.early_wait_allowed else arrival
                t = start_service + self._serv[b]
                load -= self._dem[b]
            else:
                t = arrival
        return liters, feasible

    def _make_solution(self, routes, vehicles, pending) -> Solution:
        served, liters, all_feasible = [], 0.0, True
        for vi, route in enumerate(routes):
            served.extend(route[1:-1])
            lit, feas = self._route_eval(route, vehicles[vi].start_time,
                                         vehicles[vi].capacity)
            liters += lit
            all_feasible = all_feasible and feas
        unserved = [c for c in pending if c not in set(served)]
        cost = liters + len(unserved) * self.cfg.penalty_capacity
        if not all_feasible:
            cost += self.cfg.penalty_time_window
        return Solution(routes, served, unserved, liters,
                        all_feasible and not unserved, cost)

    # ------------------------------------------------------------------
    # Local search: 2-opt + Or-opt (trong tuyến) + relocate (giữa tuyến)
    # ------------------------------------------------------------------
    def _local_search(self, sol: Solution, vehicles, pending,
                      level: str = "full") -> Solution:
        routes = [r[:] for r in sol.routes]
        passes = self.cfg.ls_max_pass if level == "intra" else self.cfg.ls_max_pass + 4
        for _ in range(passes):
            improved = False
            for vi in range(len(routes)):
                if self._improve_route(routes, vi, vehicles[vi]):
                    improved = True
            if level == "full":
                if self._relocate(routes, vehicles):
                    improved = True
                if self._inter_swap(routes, vehicles):
                    improved = True
                if self._two_opt_star(routes, vehicles):
                    improved = True
            if not improved:
                break
        return self._make_solution(routes, vehicles, pending)

    def _improve_route(self, routes, vi, veh) -> bool:
        """2-opt + Or-opt trong một tuyến; nhận nếu giảm nhiên liệu & vẫn khả thi."""
        route = routes[vi]
        if len(route) <= 3:
            return False
        best_lit, feas = self._route_eval(route, veh.start_time, veh.capacity)
        improved_any = False
        improved = True
        while improved:
            improved = False
            # 2-opt: đảo đoạn [i..j] (không đụng node xuất phát & depot cuối).
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    cand = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    lit, ok = self._route_eval(cand, veh.start_time, veh.capacity)
                    if ok and lit < best_lit - 1e-9:
                        route[:] = cand
                        best_lit = lit
                        improved = improved_any = True
            # Or-opt: dời chuỗi 1..3 khách hàng tới vị trí khác trong tuyến.
            for seg in (1, 2, 3):
                for i in range(1, len(route) - 1 - seg + 1):
                    chain = route[i:i + seg]
                    rest = route[:i] + route[i + seg:]
                    for k in range(1, len(rest) - 0):
                        if k == i:
                            continue
                        cand = rest[:k] + chain + rest[k:]
                        if cand[0] != route[0] or cand[-1] != self.depot:
                            continue
                        lit, ok = self._route_eval(cand, veh.start_time, veh.capacity)
                        if ok and lit < best_lit - 1e-9:
                            route[:] = cand
                            best_lit = lit
                            improved = improved_any = True
        return improved_any

    def _relocate(self, routes, vehicles) -> bool:
        """Dời một khách hàng sang tuyến khác nếu giảm tổng nhiên liệu."""
        improved = False
        for src in range(len(routes)):
            for pos in range(1, len(routes[src]) - 1):
                cust = routes[src][pos]
                base_src, _ = self._route_eval(routes[src], vehicles[src].start_time,
                                               vehicles[src].capacity)
                new_src = routes[src][:pos] + routes[src][pos + 1:]
                src_lit, _ = self._route_eval(new_src, vehicles[src].start_time,
                                              vehicles[src].capacity)
                for dst in range(len(routes)):
                    if dst == src:
                        continue
                    best_gain, best_insert = 0.0, None
                    base_dst, _ = self._route_eval(routes[dst],
                                                   vehicles[dst].start_time,
                                                   vehicles[dst].capacity)
                    for k in range(1, len(routes[dst])):
                        cand = routes[dst][:k] + [cust] + routes[dst][k:]
                        dst_lit, ok = self._route_eval(cand, vehicles[dst].start_time,
                                                       vehicles[dst].capacity)
                        if not ok:
                            continue
                        gain = (base_src + base_dst) - (src_lit + dst_lit)
                        if gain > best_gain + 1e-9:
                            best_gain, best_insert = gain, cand
                    if best_insert is not None:
                        routes[src] = new_src
                        routes[dst] = best_insert
                        improved = True
                        break  # tuyến src đã đổi, sang khách khác
                else:
                    continue
                break
        return improved

    def _pair_eval(self, ra, rb, va, vb):
        la, oka = self._route_eval(ra, va.start_time, va.capacity)
        lb, okb = self._route_eval(rb, vb.start_time, vb.capacity)
        return la + lb, (oka and okb)

    def _inter_swap(self, routes, vehicles) -> bool:
        """Hoán đổi một khách hàng giữa hai tuyến nếu giảm tổng nhiên liệu."""
        improved = False
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                ra, rb = routes[a], routes[b]
                if len(ra) <= 2 or len(rb) <= 2:
                    continue
                base, _ = self._pair_eval(ra, rb, vehicles[a], vehicles[b])
                done = False
                for i in range(1, len(ra) - 1):
                    for j in range(1, len(rb) - 1):
                        na, nb = ra[:], rb[:]
                        na[i], nb[j] = rb[j], ra[i]
                        cost, ok = self._pair_eval(na, nb, vehicles[a], vehicles[b])
                        if ok and cost < base - 1e-9:
                            routes[a], routes[b] = na, nb
                            improved = done = True
                            break
                    if done:
                        break
        return improved

    def _two_opt_star(self, routes, vehicles) -> bool:
        """2-opt*: tráo phần ĐUÔI của hai tuyến (gộp/giãn tải giữa các xe) — giúp
        gom xe và giảm tổng nhiên liệu. Cho phép làm rỗng một tuyến."""
        improved = False
        for a in range(len(routes)):
            for b in range(len(routes)):
                if a == b:
                    continue
                ra, rb = routes[a], routes[b]
                base, _ = self._pair_eval(ra, rb, vehicles[a], vehicles[b])
                done = False
                for i in range(0, len(ra) - 1):
                    for j in range(0, len(rb) - 1):
                        na = ra[:i + 1] + rb[j + 1:]   # đầu A + đuôi B (kết ở depot)
                        nb = rb[:j + 1] + ra[i + 1:]   # đầu B + đuôi A
                        cost, ok = self._pair_eval(na, nb, vehicles[a], vehicles[b])
                        if ok and cost < base - 1e-9:
                            routes[a], routes[b] = na, nb
                            improved = done = True
                            break
                    if done:
                        break
        return improved

    # ------------------------------------------------------------------
    def _update_pheromone(self, tau, best: Solution) -> None:
        tau *= (1.0 - self.cfg.rho)
        deposit = 1.0 / (best.cost + 1e-9)
        for route in best.routes:
            for a, b in zip(route[:-1], route[1:]):
                tau[a][b] += deposit
                tau[b][a] += deposit
        np.clip(tau, self.cfg.tau_min, self.cfg.tau_max, out=tau)
