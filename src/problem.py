"""Định nghĩa dữ liệu bài toán CVRPTW và bộ sinh instance từ đồ thị thật.

Quy ước chỉ số: node index 0 luôn là DEPOT; các khách hàng đánh số 1..N
theo đúng thứ tự trong ma trận thời gian/khoảng cách.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import ProblemConfig
from src import graph_loader


@dataclass
class Order:
    """Một đơn hàng / khách hàng cần giao."""
    cid: int                 # chỉ số trong ma trận (1..N)
    node: int                # id node trên đồ thị (để vẽ)
    demand: float            # khối lượng cần giao
    tw_start: float          # mở cửa sổ thời gian (giây)
    tw_end: float            # đóng cửa sổ thời gian (giây)
    service_time: float      # thời gian dừng giao
    release_time: float = 0.0  # thời điểm đơn được biết tới (>0 nếu đơn động)
    served: bool = False     # đã giao xong chưa (dùng trong mô phỏng động)


@dataclass
class Vehicle:
    vid: int
    capacity: float
    # Trạng thái động (dùng trong dynamic_sim):
    current_index: int = 0        # đang ở node-index nào (0 = depot)
    available_time: float = 0.0   # rảnh từ thời điểm nào
    load: float = 0.0             # tải hiện tại


@dataclass
class ProblemInstance:
    depot_node: int
    orders: List[Order]
    vehicles: List[Vehicle]
    time_matrix: np.ndarray      # (N+1) x (N+1), index 0 = depot
    dist_matrix: np.ndarray
    node_ids: List[int]          # node-index -> id node đồ thị
    coords: np.ndarray           # (N+1) x 2, toạ độ (x, y) để vẽ
    cfg: ProblemConfig
    G: object = field(default=None, repr=False)
    used_real_map: bool = False

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    def demands(self) -> np.ndarray:
        d = np.zeros(self.num_nodes)
        for o in self.orders:
            d[o.cid] = o.demand
        return d


def generate_instance(map_cfg, prob_cfg: ProblemConfig,
                      num_dynamic: int = 0, dynamic_seed: int = 7) -> ProblemInstance:
    """Sinh một instance CVRPTW: tải bản đồ, chọn node, gán demand & cửa sổ TG.

    `num_dynamic` đơn cuối được đánh dấu là "đơn phát sinh" với release_time > 0
    (chỉ được biết tới trong ca). Chúng vẫn là các node-index RIÊNG BIỆT trong
    ma trận để solver xử lý đồng nhất, chỉ khác ở chỗ bị ẩn cho tới release_time.
    """
    rng = np.random.default_rng(prob_cfg.seed)
    drng = np.random.default_rng(dynamic_seed)
    G, used_real = graph_loader.load_or_build_graph(map_cfg)

    total = prob_cfg.num_customers + num_dynamic
    # Chọn (total + 1) node: phần tử đầu làm depot.
    nodes = graph_loader.sample_nodes(G, total + 1, rng)
    time_m, dist_m = graph_loader.compute_matrices(G, nodes)

    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes], dtype=float)

    latest_start = max(0.0, prob_cfg.horizon_s - prob_cfg.tw_width_s)
    orders: List[Order] = []
    for cid in range(1, len(nodes)):
        is_dynamic = cid > prob_cfg.num_customers
        demand = float(rng.uniform(prob_cfg.demand_min, prob_cfg.demand_max))
        if is_dynamic:
            # Đơn động: xuất hiện ngẫu nhiên trong ca, cửa sổ TG tính từ lúc đó.
            release = float(drng.uniform(0.1 * prob_cfg.horizon_s, 0.7 * prob_cfg.horizon_s))
            tw_start = release
            tw_end = min(prob_cfg.horizon_s, release + prob_cfg.tw_width_s)
        else:
            release = 0.0
            tw_start = float(rng.uniform(0.0, latest_start))
            tw_end = tw_start + prob_cfg.tw_width_s
        orders.append(Order(
            cid=cid,
            node=nodes[cid],
            demand=demand,
            tw_start=tw_start,
            tw_end=tw_end,
            service_time=prob_cfg.service_time_s,
            release_time=release,
        ))

    vehicles = [Vehicle(vid=k, capacity=prob_cfg.vehicle_capacity)
                for k in range(prob_cfg.num_vehicles)]

    return ProblemInstance(
        depot_node=nodes[0],
        orders=orders,
        vehicles=vehicles,
        time_matrix=time_m,
        dist_matrix=dist_m,
        node_ids=nodes,
        coords=coords,
        cfg=prob_cfg,
        G=G,
        used_real_map=used_real,
    )


def instance_from_nodes(G, node_ids: List[int], prob_cfg: ProblemConfig,
                        demands: Optional[List[float]] = None,
                        used_real: bool = True) -> ProblemInstance:
    """Dựng instance từ danh sách NÚT cụ thể do người dùng chọn trên bản đồ.
    Phần tử đầu của `node_ids` là KHO. Cửa sổ thời gian để rộng toàn ca (người
    dùng chỉ muốn xem định tuyến); khối lượng lấy theo `demands` hoặc ngẫu nhiên."""
    rng = np.random.default_rng(prob_cfg.seed)
    time_m, dist_m = graph_loader.compute_matrices(G, node_ids)
    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids], dtype=float)

    orders: List[Order] = []
    for cid in range(1, len(node_ids)):
        if demands is not None and cid - 1 < len(demands) and demands[cid - 1]:
            dem = float(demands[cid - 1])
        else:
            dem = float(rng.uniform(prob_cfg.demand_min, prob_cfg.demand_max))
        orders.append(Order(
            cid=cid, node=node_ids[cid], demand=dem,
            tw_start=0.0, tw_end=prob_cfg.horizon_s,
            service_time=prob_cfg.service_time_s, release_time=0.0))

    vehicles = [Vehicle(vid=k, capacity=prob_cfg.vehicle_capacity)
                for k in range(prob_cfg.num_vehicles)]
    return ProblemInstance(
        depot_node=node_ids[0], orders=orders, vehicles=vehicles,
        time_matrix=time_m, dist_matrix=dist_m, node_ids=node_ids, coords=coords,
        cfg=prob_cfg, G=G, used_real_map=used_real)


def known_orders(inst: ProblemInstance, now: float) -> List[Order]:
    """Các đơn đã được biết tới tại thời điểm `now` và chưa giao xong."""
    return [o for o in inst.orders if o.release_time <= now and not o.served]
