"""Các thuật toán baseline để so sánh công bằng với hybrid ACO.

  1. nearest_neighbor  : heuristic xây dựng tham lam (chọn điểm gần nhất khả thi).
  2. nearest_neighbor + 2-opt : NN rồi tinh chỉnh bằng 2-opt — baseline mạnh, luôn sẵn.
  3. or_tools_solve    : Google OR-Tools (nếu đã cài) — lời giải tham chiếu chất lượng cao.

Mọi baseline trả về `Solution` cùng kiểu với ACO để dùng chung metrics.evaluate.
Hàm mục tiêu đánh giá vẫn là nhiên liệu PRP (qua solver._route_eval) để so sánh
trên cùng một thước đo.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from config import ACOConfig
from src.aco_solver import ACOSolver, Solution, VehicleState
from src.emission_model import EmissionModel


def _eval_solution(solver: ACOSolver, routes, vehicles, pending) -> Solution:
    return solver._make_solution(routes, vehicles, pending)


def nearest_neighbor(pending: Sequence[int], vehicles: Sequence[VehicleState],
                     solver: ACOSolver) -> Solution:
    """Xây tuyến tham lam: mỗi xe liên tục đi tới khách hàng KHẢ THI gần nhất
    (theo thời gian có tắc) cho tới khi hết khả thi, rồi về kho."""
    unvisited = set(pending)
    routes: List[List[int]] = []
    for v in vehicles:
        route = [v.start_index]
        cur, t, load = v.start_index, v.start_time, 0.0
        while unvisited:
            cand = solver._feasible(cur, t, load, v.capacity, unvisited)
            if not cand:
                break
            nxt = min(cand, key=lambda j: solver._t[cur][j])
            eff = solver._t[cur][nxt] * solver.cong(t)
            arrival = t + eff
            t = max(arrival, solver._tw_s[nxt]) + solver._serv[nxt]
            load += solver._dem[nxt]
            cur = nxt
            route.append(nxt)
            unvisited.discard(nxt)
        route.append(solver.depot)
        routes.append(route)
    return _eval_solution(solver, routes, vehicles, pending)


def nearest_neighbor_2opt(pending, vehicles, solver: ACOSolver) -> Solution:
    """NN rồi áp 2-opt/Or-opt TRONG TUYẾN (đúng nghĩa 'NN+2-opt' kinh điển).
    Không dùng phép liên tuyến để baseline này phân biệt rõ với Hybrid ACO."""
    sol = nearest_neighbor(pending, vehicles, solver)
    return solver._local_search(sol, vehicles, pending, level="intra")


def or_tools_solve(pending, vehicles, solver: ACOSolver) -> Optional[Solution]:
    """Giải bằng Google OR-Tools nếu khả dụng (tối thiểu hoá thời gian có tắc
    tại giờ khởi hành đầu ca — xấp xỉ). Trả về None nếu chưa cài OR-Tools."""
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None

    n = solver._t.shape[0]
    starts = [v.start_index for v in vehicles]
    ends = [solver.depot for _ in vehicles]
    mgr = pywrapcp.RoutingIndexManager(n, len(vehicles), starts, ends)
    routing = pywrapcp.RoutingModel(mgr)

    cong0 = solver.cong(vehicles[0].start_time if vehicles else 0.0)
    t_int = (solver._t * cong0).astype(int)

    def cb(i, j):
        return int(t_int[mgr.IndexToNode(i)][mgr.IndexToNode(j)])

    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)

    # Ràng buộc tải trọng.
    dem_int = solver._dem.astype(int)

    def dcb(i):
        node = mgr.IndexToNode(i)
        return int(dem_int[node]) if node in set(pending) else 0

    didx = routing.RegisterUnaryTransitCallback(dcb)
    routing.AddDimensionWithVehicleCapacity(
        didx, 0, [int(v.capacity) for v in vehicles], True, "Cap")

    # Ràng buộc CỬA SỔ THỜI GIAN: thời gian cộng dồn = di chuyển + phục vụ;
    # cho phép xe chờ (slack) tới hết ca. Đây là phần làm OR-Tools thành baseline
    # VRPTW đúng nghĩa (nếu thiếu, lời giải sẽ ngắn giả tạo do bỏ qua cửa sổ TG).
    serv_int = solver._serv.astype(int)
    horizon = int(solver._horizon)

    def tcb(i, j):
        fn = mgr.IndexToNode(i)
        return int(t_int[fn][mgr.IndexToNode(j)] + serv_int[fn])

    tidx = routing.RegisterTransitCallback(tcb)
    routing.AddDimension(tidx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    tw_s = solver._tw_s.astype(int)
    tw_e = solver._tw_e.astype(int)
    for node in range(n):
        index = mgr.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(int(tw_s[node]), int(tw_e[node]))
    for k in range(len(vehicles)):
        time_dim.CumulVar(routing.Start(k)).SetRange(0, horizon)

    # Chỉ buộc phục vụ các node trong `pending`; node khác cho phép bỏ.
    pend = set(pending)
    for node in range(n):
        if node not in pend and node != solver.depot:
            routing.AddDisjunction([mgr.NodeToIndex(node)], 0)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = 5

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    routes: List[List[int]] = []
    for k in range(len(vehicles)):
        idx = routing.Start(k)
        route = [mgr.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            idx = sol.Value(routing.NextVar(idx))
            route.append(mgr.IndexToNode(idx))
        routes.append(route)
    return _eval_solution(solver, routes, vehicles, pending)
