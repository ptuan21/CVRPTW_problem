"""Tham số cấu hình tập trung cho toàn bộ dự án định tuyến động.

Mọi hằng số "ma thuật" được đặt ở đây để tiện tinh chỉnh
(ablation study). Các nhóm tham số: bản đồ, bài toán, mô hình nhiên liệu,
thuật toán ACO, và mô phỏng động.
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Bản đồ / mạng đường
# ---------------------------------------------------------------------------
@dataclass
class MapConfig:
    # Địa danh để tải bản đồ thật qua OSMnx. Có thể đổi sang "Hoan Kiem, Hanoi"...
    place: str = "Hoan Kiem District, Hanoi, Vietnam"
    network_type: str = "drive"          # chỉ lấy đường ô tô đi được
    # Nếu không tải được OSMnx (offline / chưa cài) thì dùng lưới mô phỏng:
    fallback_grid_rows: int = 12
    fallback_grid_cols: int = 12
    fallback_grid_spacing_m: float = 250.0   # khoảng cách giữa 2 nút lưới (mét)
    default_speed_kph: float = 30.0      # vận tốc mặc định khi cạnh thiếu maxspeed
    cache_path: str = "data/graph.graphml"


# ---------------------------------------------------------------------------
# 2. Bài toán giao hàng (CVRPTW)
# ---------------------------------------------------------------------------
@dataclass
class ProblemConfig:
    num_customers: int = 25              # số đơn hàng ban đầu
    num_vehicles: int = 5                # số xe trong đội
    vehicle_capacity: float = 100.0      # tải trọng tối đa mỗi xe (đơn vị hàng)
    demand_min: float = 5.0
    demand_max: float = 20.0
    service_time_s: float = 120.0        # thời gian dừng giao mỗi điểm (giây)
    # Khung thời gian (giây kể từ mốc 0 = đầu ca). Ca làm 8 giờ.
    horizon_s: float = 8 * 3600.0
    tw_width_s: float = 2 * 3600.0       # độ rộng cửa sổ thời gian mỗi đơn
    seed: int = 42                        # hạt giống để tái lập kết quả


# ---------------------------------------------------------------------------
# 3. Loại nhiên liệu (Hà Nội đã chuyển hoàn toàn sang xăng E10 từ 2026)
# ---------------------------------------------------------------------------
@dataclass
class FuelType:
    """Đặc tính một loại nhiên liệu.

    - energy_factor: hệ số tiêu thụ THỂ TÍCH so với xăng khoáng gốc (E0 = 1.00).
      Ethanol có nhiệt trị thấp hơn xăng (~21.1 vs ~32.2 MJ/L) nên xăng càng
      pha nhiều ethanol thì phải đốt nhiều lít hơn cho cùng quãng đường.
    - price: giá bán lẻ (đồng/lít) — minh hoạ, có thể cập nhật theo thực tế.
    - co2_kg_per_liter: phát thải CO2 HOÁ THẠCH mỗi lít. Phần ethanol sinh học
      được tính trung hoà carbon nên xăng pha ethanol giảm CO2 hoá thạch.
    - bio_fraction: tỉ lệ ethanol sinh học (thể tích).
    """
    name: str
    energy_factor: float
    price: float
    co2_kg_per_liter: float
    bio_fraction: float = 0.0


# Bảng tra các loại nhiên liệu phổ biến tại Việt Nam (số liệu xấp xỉ, tham số hoá).
# Nhiệt trị: xăng ~32.2 MJ/L, ethanol ~21.1 MJ/L -> energy_factor = 32.2 / LHV_pha.
# CO2 hoá thạch xăng khoáng ~2.31 kg/L; phần ethanol coi như trung hoà carbon.
FUEL_TYPES = {
    "E0":  FuelType("Xăng khoáng RON95 (E0)", energy_factor=1.000, price=21000.0,
                    co2_kg_per_liter=2.31, bio_fraction=0.00),
    "E5":  FuelType("Xăng E5 RON92",          energy_factor=1.017, price=20000.0,
                    co2_kg_per_liter=2.19, bio_fraction=0.05),
    "E10": FuelType("Xăng E10",               energy_factor=1.036, price=19500.0,
                    co2_kg_per_liter=2.08, bio_fraction=0.10),
}


@dataclass
class FuelConfig:
    """Vật lý tiêu thụ nhiên liệu — xấp xỉ tuyến tính theo tổng khối lượng.

        liters_E0_per_m = (c0 + c1 * (curb_weight + load)) / 1000
        liters_thuc_te  = liters_E0_per_m * fuel_type.energy_factor

    Ý nghĩa: chở càng nặng càng tốn xăng -> thuật toán có động lực giao các đơn
    nặng sớm để giảm tải trên các chặng sau (khác với chỉ tối thiểu quãng đường).
    `c0, c1, curb_weight` đặc trưng cho ĐỘNG CƠ/XE (quy về xăng khoáng gốc);
    loại nhiên liệu được tách riêng qua `fuel_type`.
    """
    curb_weight: float = 1500.0          # khối lượng xe rỗng (kg, quy đổi)
    c0: float = 0.06                     # tiêu thụ cơ sở E0 (lít/km) khi không tải
    c1: float = 0.00003                  # phần tăng thêm theo mỗi kg tải
    fuel_type: str = "E10"               # loại nhiên liệu mặc định (Hà Nội 2026)


# ---------------------------------------------------------------------------
# 3b. Mô hình phát thải vi mô theo TỐC ĐỘ (Pollution-Routing Problem)
# ---------------------------------------------------------------------------
@dataclass
class PRPConfig:
    """Tham số mô hình tiêu thụ nhiên liệu toàn diện (Bektaş & Laporte 2011;
    Demir et al. 2012; mô hình CMEM). Nhiên liệu trên một cung phụ thuộc:
    TỐC ĐỘ, tải trọng, gia tốc và độ dốc — nên có một tốc độ tối ưu hình chữ U
    (chậm quá tốn nhiên liệu nền động cơ, nhanh quá tốn lực cản khí động).

    Lít nhiên liệu trên cung dài d (m) chạy đều ở tốc độ v (m/s), tải `load`:
      F = λ·k·Nₑ·Vₑ·(d/v)              # mô-đun động cơ (idle/ma sát)
        + λ·γ·α·(M+load)·d             # lực kéo: gia tốc + dốc + lăn
        + λ·γ·β·d·v²                   # lực cản khí động
    với λ = ξ/(κ·ψ), γ = 1/(1000·η_tf·η), β = ½·C_d·ρ·A, α = a + g·sinθ + g·C_r·cosθ.
    """
    xi: float = 1.0          # ξ tỉ lệ khối lượng nhiên liệu/không khí
    k_eng: float = 0.2       # k hệ số ma sát động cơ (kJ/vòng/L)
    Ne: float = 33.0         # Nₑ tốc độ vòng tua động cơ (vòng/s)
    Ve: float = 2.5          # Vₑ dung tích xy-lanh (L) — xe tải nhẹ đô thị
    kappa: float = 43.2      # κ nhiệt trị nhiên liệu (kJ/g) — xăng khoáng
    psi: float = 737.0       # ψ khối lượng riêng nhiên liệu (g/L)
    eta_tf: float = 0.4      # η_tf hiệu suất truyền động
    eta_eng: float = 0.9     # η hiệu suất động cơ
    Cd: float = 0.7          # C_d hệ số cản khí động
    rho_air: float = 1.2041  # ρ khối lượng riêng không khí (kg/m³)
    area: float = 3.912      # A diện tích cản chính diện (m²)
    Cr: float = 0.01         # C_r hệ số cản lăn
    g: float = 9.81
    accel: float = 0.0       # a gia tốc trung bình (0 = chạy đều)
    road_angle: float = 0.0  # θ độ dốc (rad)
    # Giới hạn tốc độ cho phép tối ưu (m/s): ~18 km/h .. 90 km/h
    v_min: float = 5.0
    v_max: float = 25.0


# ---------------------------------------------------------------------------
# 4. Thuật toán Ant Colony (MAX-MIN Ant System)
# ---------------------------------------------------------------------------
@dataclass
class ACOConfig:
    num_ants: int = 25
    num_iterations: int = 120
    alpha: float = 1.0                   # trọng số pheromone
    beta: float = 3.0                    # trọng số heuristic (1/chi phí)
    rho: float = 0.1                     # tốc độ bay hơi pheromone
    q0: float = 0.1                      # xác suất khai thác tham lam (ACS)
    tau_max: float = 10.0                # cận trên pheromone (MMAS)
    tau_min: float = 0.05                # cận dưới pheromone (MMAS)
    # Phạt khi vi phạm ràng buộc để giữ lời giải khả thi:
    penalty_capacity: float = 1e6
    penalty_time_window: float = 1e5
    early_wait_allowed: bool = True      # cho phép xe chờ nếu đến sớm hơn cửa sổ
    # Hybrid: local search tinh chỉnh lời giải tốt nhất mỗi vòng.
    local_search: bool = True            # bật 2-opt + Or-opt
    ls_max_pass: int = 2                 # số lượt quét cải thiện tối đa


# ---------------------------------------------------------------------------
# 4b. Giao thông phụ thuộc thời điểm (Time-Dependent) — giờ cao điểm Hà Nội
# ---------------------------------------------------------------------------
@dataclass
class TrafficConfig:
    """Hệ số tắc đường theo giờ trong ngày: thời gian di chuyển thực = thời gian
    thông thoáng × hệ số(giờ khởi hành). Hai đỉnh kẹt sáng/chiều. Vì tốc độ =
    quãng đường / thời gian, giờ cao điểm -> tốc độ giảm -> phát thải tăng
    (khép kín vòng "tắc đường -> tốn nhiên liệu")."""
    enabled: bool = True
    shift_start_hour: float = 7.0        # mốc giờ ứng với thời điểm 0 của ca
    base_multiplier: float = 1.0         # hệ số lúc thông thoáng
    morning_peak_hour: float = 8.0       # đỉnh kẹt sáng
    evening_peak_hour: float = 17.5      # đỉnh kẹt chiều
    peak_multiplier: float = 2.0         # hệ số tại đỉnh kẹt
    peak_width_h: float = 1.5            # độ rộng (giờ) của mỗi đỉnh


# ---------------------------------------------------------------------------
# 5. Mô phỏng động (rolling horizon)
# ---------------------------------------------------------------------------
@dataclass
class DynamicConfig:
    num_dynamic_orders: int = 15         # số đơn phát sinh trong ca
    replan_interval_s: float = 1800.0    # chu kỳ tái tối ưu định kỳ (30 phút)
    replan_on_new_order: bool = True     # tái tối ưu ngay khi có đơn mới
    # Sự kiện tắc đường ngẫu nhiên theo vùng (chồng lên hồ sơ giờ cao điểm):
    traffic_event_prob: float = 0.3      # xác suất có tắc đường mỗi epoch
    traffic_multiplier: float = 2.2      # mức tăng thời gian khi tắc
    # Thời tiết xấu làm chậm toàn mạng:
    weather_multiplier: float = 1.0      # 1.0 = bình thường; 1.3 = mưa
    warm_start_pheromone: bool = True    # giữ pheromone giữa các epoch (tái tối ưu nhanh)
    seed: int = 7


@dataclass
class Config:
    map: MapConfig = field(default_factory=MapConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    fuel: FuelConfig = field(default_factory=FuelConfig)
    prp: PRPConfig = field(default_factory=PRPConfig)
    aco: ACOConfig = field(default_factory=ACOConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)


# Cấu hình mặc định dùng chung
CFG = Config()
