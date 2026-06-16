"""Máy chủ web nhẹ (thư viện chuẩn, không cần Flask) phục vụ giao diện bản đồ
thực để thử nghiệm thuật toán định tuyến — kiểu Google Maps bằng Leaflet + OSM.

Chạy:  python -m webapp.server      (rồi mở http://localhost:8000)

API:
  GET /                -> trang bản đồ.
  GET /api/solve?...   -> sinh instance + giải, trả JSON (depot, khách, lộ trình
                          bám đường phố, chỉ số). Tham số:
        customers, vehicles, capacity, fuel(E0|E5|E10), hour, congestion(0|1),
        mode(static|dynamic), dynamic_orders, seed.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from config import CFG, FUEL_TYPES
from src import problem, metrics, graph_loader, baseline
from src.aco_solver import ACOSolver, VehicleState
from src.emission_model import EmissionModel
from src.travel import Congestion
from src.dynamic_sim import _order_arrays, run_dynamic

HERE = os.path.dirname(__file__)
_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]

# Cache graph + instance theo cấu hình để khỏi tải lại mỗi request.
_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH, used = graph_loader.load_or_build_graph(CFG.map)
        print(f"[web] Đã tải graph (bản đồ thật={used}).")
    return _GRAPH


def _build_config(g, mode):
    cfg = copy.deepcopy(CFG)
    cust = g("customers", 25)
    cfg.problem.num_customers = 25 if isinstance(cust, list) else int(cust)
    cfg.problem.num_vehicles = int(g("vehicles", 5))
    cfg.problem.vehicle_capacity = float(g("capacity", 100))
    cfg.problem.seed = int(g("seed", 42))
    cfg.fuel.fuel_type = g("fuel", "E10")
    cfg.traffic.enabled = str(g("congestion", "1")) in ("1", "true", "True")
    cfg.traffic.shift_start_hour = float(g("hour", 8))
    cfg.dynamic.num_dynamic_orders = int(g("dynamic_orders", 15))
    # Cấu hình ACO NHẸ cho giao diện (phản hồi nhanh); thí nghiệm vẫn dùng mặc định.
    if mode == "dynamic":
        cfg.aco.num_ants = 18
        cfg.aco.num_iterations = 25   # warm-start nên hội tụ nhanh
    else:
        cfg.aco.num_ants = 20
        cfg.aco.num_iterations = 60
    return cfg


def _node_latlon(inst, idx):
    n = inst.node_ids[idx]
    return [float(inst.G.nodes[n]["y"]), float(inst.G.nodes[n]["x"])]


def _street_polyline(inst, a_idx, b_idx):
    return graph_loader.path_coords(inst.G, inst.node_ids[a_idx], inst.node_ids[b_idx])


def solve_payload(params):
    """params: dict phẳng. Hỗ trợ 2 nguồn điểm:
      - ngẫu nhiên (mặc định): sinh từ bản đồ;
      - tự chọn: params['depot'] = {lat,lon} và params['customers'] = [{lat,lon,demand}].
    """
    def g(key, default=None):
        v = params.get(key, default)
        if isinstance(v, list) and v and not isinstance(v[0], dict):
            return v[0]  # parse_qs trả list
        return v

    mode = g("mode", "static")
    algo = g("algo", "aco")
    cfg = _build_config(g, mode)
    num_dyn = cfg.dynamic.num_dynamic_orders if mode == "dynamic" else 0

    G = _get_graph()
    used_real = _is_real(G)
    manual_depot = params.get("depot")
    manual_custs = params.get("customers")

    if manual_depot and manual_custs:
        # Tự chọn: snap từng điểm vào nút đường gần nhất.
        node_ids = [graph_loader.nearest_node(G, float(manual_depot["lat"]),
                                              float(manual_depot["lon"]))]
        demands = []
        for c in manual_custs:
            node_ids.append(graph_loader.nearest_node(G, float(c["lat"]), float(c["lon"])))
            demands.append(float(c.get("demand") or 0))
        cfg.problem.num_customers = len(manual_custs)
        inst = problem.instance_from_nodes(G, node_ids, cfg.problem,
                                           demands=demands, used_real=used_real)
        num_dyn = 0
        mode = "static"  # tự chọn chỉ ở chế độ tĩnh
    else:
        # Ngẫu nhiên: dùng graph đã cache qua monkeypatch nhẹ.
        orig = graph_loader.load_or_build_graph
        graph_loader.load_or_build_graph = lambda _c: (G, used_real)
        try:
            inst = problem.generate_instance(cfg.map, cfg.problem,
                                             num_dynamic=num_dyn,
                                             dynamic_seed=cfg.dynamic.seed)
        finally:
            graph_loader.load_or_build_graph = orig

    em = EmissionModel(cfg.prp, cfg.fuel)
    cong = Congestion(cfg.traffic)

    depot = _node_latlon(inst, 0)
    customers = [{"id": o.cid, "lat": _node_latlon(inst, o.cid)[0],
                  "lon": _node_latlon(inst, o.cid)[1], "demand": round(o.demand, 1),
                  "dynamic": o.release_time > 0,
                  "tw": [round(o.tw_start / 3600, 1), round(o.tw_end / 3600, 1)]}
                 for o in inst.orders]

    if mode == "dynamic":
        routes_json, summary = _solve_dynamic(inst, cfg, em)
    else:
        routes_json, summary = _solve_static(inst, cfg, em, cong, algo)

    return {
        "used_real_map": inst.used_real_map,
        "center": depot,
        "depot": {"lat": depot[0], "lon": depot[1]},
        "customers": customers,
        "routes": routes_json,
        "metrics": summary,
        "fuel_name": em.fuel.name,
        "optimal_speed_kph": round(em.optimal_speed() * 3.6, 1),
        "mode": mode,
        "algo": algo,
    }


def _solve_static(inst, cfg, em, cong, algo="aco"):
    demand, tw_s, tw_e, serv = _order_arrays(inst)
    pending = [o.cid for o in inst.orders]
    states = [VehicleState(vid=k, start_index=0, start_time=0.0,
                           capacity=cfg.problem.vehicle_capacity)
              for k in range(cfg.problem.num_vehicles)]
    solver = ACOSolver(cfg.aco, em, congestion=cong, depot_index=0,
                       rng=np.random.default_rng(1))
    solver.bind(inst.time_matrix, inst.dist_matrix, demand, tw_s, tw_e, serv,
                cfg.problem.horizon_s)
    if algo == "ortools":
        sol = baseline.or_tools_solve(pending, states, solver)
        if sol is None:  # chưa cài OR-Tools -> rơi về ACO
            algo = "aco"
    if algo == "nn2opt":
        sol = baseline.nearest_neighbor_2opt(pending, states, solver)
    if algo == "aco":
        sol, _, _ = solver.solve(pending, states, inst.time_matrix, inst.dist_matrix,
                                 demand, tw_s, tw_e, serv, cfg.problem.horizon_s)
    m = metrics.evaluate(sol, inst.dist_matrix, inst.time_matrix, demand, tw_s,
                         tw_e, serv, em, congestion=cong,
                         horizon=cfg.problem.horizon_s,
                         vehicle_start_times=[0.0] * cfg.problem.num_vehicles)

    routes_json = []
    for vi, route in enumerate(sol.routes):
        if len(route) <= 2:
            continue
        poly = []
        dist = 0.0
        for a, b in zip(route[:-1], route[1:]):
            poly.extend(_street_polyline(inst, a, b))
            dist += inst.dist_matrix[a][b]
        lit, _ = solver._route_eval(route, 0.0, cfg.problem.vehicle_capacity)
        routes_json.append({
            "vehicle": vi,
            "color": _COLORS[len(routes_json) % len(_COLORS)],
            "stops": [int(c) for c in route],
            "polyline": poly,
            "num_stops": len(route) - 2,
            "liters": round(lit, 2),
            "distance_km": round(dist / 1000.0, 1),
        })
    summary = {
        "served": m.served, "unserved": m.unserved,
        "liters": round(m.fuel_liters, 2), "cost": round(m.fuel_cost),
        "co2": round(m.co2_kg, 2), "vehicles_used": m.num_vehicles_used,
        "avg_speed": round(m.avg_speed_kph, 1),
        "distance_km": round(m.total_distance_km, 2),
        "tw_violations": m.tw_violations,
    }
    return routes_json, summary


def _solve_dynamic(inst, cfg, em):
    res = run_dynamic(inst, cfg)
    # Gom các chặng đã thực thi theo xe.
    by_v = {}
    for leg in res.executed_legs:
        by_v.setdefault(leg.vid, []).append(leg)
    routes_json = []
    for vi, legs in sorted(by_v.items()):
        poly = []
        dist = sum(leg.dist for leg in legs)
        for leg in legs:
            poly.extend(_street_polyline(inst, leg.a, leg.b))
        routes_json.append({
            "vehicle": vi,
            "color": _COLORS[len(routes_json) % len(_COLORS)],
            "stops": [],
            "polyline": poly,
            "num_stops": len(legs),
            "liters": None,
            "distance_km": round(dist / 1000.0, 1),
        })
    summary = {
        "served": res.served, "unserved": res.unserved,
        "liters": round(res.fuel_liters, 2), "cost": round(res.fuel_cost),
        "co2": round(res.co2_kg, 2), "vehicles_used": len(routes_json),
        "avg_speed": None, "distance_km": round(res.total_distance_km, 2),
        "num_replans": res.num_replans,
        "avg_solve_ms": round(res.avg_solve_time_s * 1000),
    }
    return routes_json, summary


def _is_real(G):
    # Bản đồ thật của OSMnx có toạ độ kinh độ ~100..110 cho VN; lưới fallback ~0..vài nghìn m.
    try:
        xs = [G.nodes[n]["x"] for n in list(G.nodes)[:5]]
        return any(abs(x) > 50 for x in xs)
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/solve":
            try:
                payload = solve_payload(parse_qs(parsed.query))
                self._send(200, json.dumps(payload))
            except Exception as exc:  # trả lỗi gọn cho frontend
                import traceback
                traceback.print_exc()
                self._send(500, json.dumps({"error": str(exc)}))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/solve":
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, json.dumps(solve_payload(body)))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._send(500, json.dumps({"error": str(exc)}))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *args):
        pass  # tắt log ồn ào


def main(port=8000):
    _get_graph()
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[web] Mở giao diện tại  http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(p)
