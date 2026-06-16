"""Tải mạng đường và tính ma trận thời gian/khoảng cách giữa các điểm.

Hai chế độ:
  1. Bản đồ THẬT qua OSMnx (OpenStreetMap) — ưu tiên, dùng cho báo cáo.
  2. Lưới mô phỏng (fallback) khi offline / chưa cài OSMnx — để code luôn chạy.

Đầu ra dùng cho solver là các ma trận numpy:
  - time_matrix[i][j]  : thời gian di chuyển (giây) theo đường ngắn nhất
  - dist_matrix[i][j]  : quãng đường (mét) theo đường ngắn nhất
trong đó i, j là chỉ số trong danh sách node được chọn (depot + khách).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import networkx as nx
import numpy as np

from config import MapConfig


# ---------------------------------------------------------------------------
# Tải / xây dựng đồ thị nền
# ---------------------------------------------------------------------------
def load_or_build_graph(cfg: MapConfig):
    """Trả về (G, used_real_map). G là networkx graph có thuộc tính cạnh
    'length' (mét) và 'travel_time' (giây)."""
    # Ưu tiên cache để khỏi gọi mạng nhiều lần.
    if os.path.exists(cfg.cache_path):
        try:
            import osmnx as ox
            G = ox.load_graphml(cfg.cache_path)
            return _ensure_edge_attrs(G, cfg), True
        except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
            print(f"[graph] Không đọc được cache ({exc}); tải lại.")

    try:
        import osmnx as ox

        print(f"[graph] Tải bản đồ thật từ OSM: {cfg.place} ...")
        G = ox.graph_from_place(cfg.place, network_type=cfg.network_type)
        # Bổ sung vận tốc + thời gian di chuyển dựa trên maxspeed của OSM.
        G = ox.add_edge_speeds(G, fallback=cfg.default_speed_kph)
        G = ox.add_edge_travel_times(G)
        os.makedirs(os.path.dirname(cfg.cache_path) or ".", exist_ok=True)
        ox.save_graphml(G, cfg.cache_path)
        return _ensure_edge_attrs(G, cfg), True
    except Exception as exc:
        print(f"[graph] OSMnx không khả dụng ({exc}); dùng lưới mô phỏng.")
        return build_grid_graph(cfg), False


def _ensure_edge_attrs(G, cfg: MapConfig):
    """Đảm bảo mọi cạnh có 'length' và 'travel_time' hợp lệ."""
    default_mps = cfg.default_speed_kph / 3.6
    for _, _, data in G.edges(data=True):
        length = float(data.get("length", 0.0) or 0.0)
        data["length"] = length
        tt = data.get("travel_time")
        if tt is None or tt <= 0:
            data["travel_time"] = length / default_mps if default_mps else length
        else:
            data["travel_time"] = float(tt)
    return G


def build_grid_graph(cfg: MapConfig):
    """Lưới đường ô bàn cờ — fallback khi không có bản đồ thật.

    Mỗi nút có toạ độ (x, y) tính bằng mét; cạnh hai chiều với độ dài =
    khoảng cách Euclid, travel_time theo vận tốc mặc định.
    """
    rows, cols = cfg.fallback_grid_rows, cfg.fallback_grid_cols
    s = cfg.fallback_grid_spacing_m
    speed_mps = cfg.default_speed_kph / 3.6

    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:3857"  # toạ độ phẳng (mét) cho tiện vẽ
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            G.add_node(n, x=c * s, y=r * s)

    def connect(a, b):
        dx = G.nodes[a]["x"] - G.nodes[b]["x"]
        dy = G.nodes[a]["y"] - G.nodes[b]["y"]
        d = float((dx * dx + dy * dy) ** 0.5)
        for u, v in ((a, b), (b, a)):
            G.add_edge(u, v, length=d, travel_time=d / speed_mps)

    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                connect(n, n + 1)
            if r + 1 < rows:
                connect(n, n + cols)
    return G


# ---------------------------------------------------------------------------
# Chọn node và tính ma trận đường ngắn nhất
# ---------------------------------------------------------------------------
def sample_nodes(G, n: int, rng: np.random.Generator) -> List[int]:
    """Chọn ngẫu nhiên n node (kèm depot ở đầu) từ thành phần liên thông lớn nhất."""
    # Lấy thành phần liên thông mạnh lớn nhất để đảm bảo mọi cặp đều có đường đi.
    if G.is_directed():
        comp = max(nx.strongly_connected_components(G), key=len)
    else:
        comp = max(nx.connected_components(G), key=len)
    nodes = list(comp)
    idx = rng.choice(len(nodes), size=min(n, len(nodes)), replace=False)
    return [nodes[i] for i in idx]


def compute_matrices(G, nodes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Ma trận thời gian (giây) và khoảng cách (mét) giữa mọi cặp node đã chọn.

    Dùng Dijkstra một-nguồn nhiều-đích theo 'travel_time' và cộng dồn 'length'
    của đường ngắn nhất theo thời gian (gần với hành vi tài xế thực tế).
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    target_set = set(nodes)
    time_m = np.full((n, n), np.inf)
    dist_m = np.full((n, n), np.inf)

    for src in nodes:
        # Dijkstra theo travel_time, trả về cả đường đi để cộng length.
        lengths, paths = nx.single_source_dijkstra(G, src, weight="travel_time")
        i = index[src]
        for dst in target_set:
            if dst not in paths:
                continue
            j = index[dst]
            time_m[i][j] = lengths[dst]
            dist_m[i][j] = _path_length(G, paths[dst])
    np.fill_diagonal(time_m, 0.0)
    np.fill_diagonal(dist_m, 0.0)
    _patch_unreachable(time_m, dist_m)
    return time_m, dist_m


def nearest_node(G, lat: float, lon: float) -> int:
    """Nút đường gần nhất với toạ độ (lat, lon) người dùng nhấp trên bản đồ.

    Dùng khoảng cách Euclid trên (kinh độ, vĩ độ) — đủ chính xác ở phạm vi đô thị."""
    best, best_d = None, float("inf")
    for n, data in G.nodes(data=True):
        dy = data["y"] - lat
        dx = data["x"] - lon
        d = dx * dx + dy * dy
        if d < best_d:
            best_d, best = d, n
    return best


def path_coords(G, node_a: int, node_b: int) -> List[List[float]]:
    """Toạ độ [lat, lon] dọc đường ngắn nhất (theo thời gian) giữa hai node, BÁM
    SÁT HÌNH HỌC ĐƯỜNG PHỐ. Mỗi cạnh OSM cong có thuộc tính `geometry` (LineString)
    chứa các điểm uốn; nếu chỉ nối toạ độ nút giao bằng đường thẳng thì sẽ "cắt
    góc" trông như đi xuyên nhà — nên ở đây ta dùng `geometry` khi có."""
    try:
        path = nx.shortest_path(G, node_a, node_b, weight="travel_time")
    except Exception:
        path = [node_a, node_b]

    out: List[List[float]] = []
    for u, v in zip(path[:-1], path[1:]):
        edges = G.get_edge_data(u, v)
        pts = None
        if edges:
            data = min(edges.values(), key=lambda d: d.get("travel_time", float("inf")))
            geom = data.get("geometry")
            if geom is not None:
                # LineString.coords trả [(lon, lat), ...]; đổi sang [lat, lon].
                pts = [[float(y), float(x)] for x, y in geom.coords]
                # Đảm bảo hướng từ u -> v.
                uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
                if pts and (abs(pts[0][0] - uy) + abs(pts[0][1] - ux)
                            > abs(pts[-1][0] - uy) + abs(pts[-1][1] - ux)):
                    pts.reverse()
        if pts is None:
            pts = [[float(G.nodes[u]["y"]), float(G.nodes[u]["x"])],
                   [float(G.nodes[v]["y"]), float(G.nodes[v]["x"])]]
        if out and pts and out[-1] == pts[0]:
            pts = pts[1:]  # tránh lặp nút chung
        out.extend(pts)
    return out


def _path_length(G, path: List[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        # Với MultiDiGraph lấy cạnh có length nhỏ nhất giữa u, v.
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", np.inf))
        total += float(data.get("length", 0.0))
    return total


def _patch_unreachable(time_m: np.ndarray, dist_m: np.ndarray) -> None:
    """Thay các cặp không tới được (inf) bằng giá trị phạt lớn nhưng hữu hạn
    để solver không sụp đổ; thực tế các node đã nằm chung thành phần liên thông
    nên trường hợp này hiếm."""
    finite = time_m[np.isfinite(time_m)]
    big_t = (finite.max() * 10) if finite.size else 1e6
    finite_d = dist_m[np.isfinite(dist_m)]
    big_d = (finite_d.max() * 10) if finite_d.size else 1e6
    time_m[~np.isfinite(time_m)] = big_t
    dist_m[~np.isfinite(dist_m)] = big_d
