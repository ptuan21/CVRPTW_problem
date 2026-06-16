# Định tuyến động & gom hàng thông minh nhận biết phát thải (E10)

Bài toán giao hàng đô thị có **đơn phát sinh đột xuất, tắc đường, giờ cao điểm**,
tối ưu **nhiên liệu & CO₂** theo bối cảnh **Hà Nội chuyển sang xăng E10 (2026)**.

Lõi thuật toán: **Hybrid Ant Colony Optimization** (MAX-MIN Ant System + local
search) trên nền **Time-Dependent VRPTW** với hàm mục tiêu là **mô hình phát thải
phụ thuộc tốc độ (Pollution-Routing Problem)**. Dữ liệu bản đồ **thật** từ
OpenStreetMap (OSMnx).

## Điểm nổi bật học thuật
- **Mô hình phát thải PRP theo tốc độ** (Bektaş & Laporte 2011; CMEM): nhiên liệu
  phụ thuộc tốc độ–tải–gia tốc–độ dốc, có **tốc độ tối ưu** v\* ≈ 44 km/h.
- **Giao thông phụ thuộc thời điểm**: hệ số tắc theo giờ cao điểm → tốc độ giảm →
  nhiên liệu tăng (đo được **+16.6%** trên cùng lộ trình khi vào giờ kẹt).
- **Loại nhiên liệu E0/E5/E10**: E10 tốn thêm ~3.6% thể tích nhưng **rẻ hơn ~3.8%**
  và **giảm ~6.7% CO₂ hoá thạch**.
- **Định tuyến động**: rolling-horizon, đơn phát sinh, warm-start pheromone.
- **Benchmark**: so với Nearest-Neighbor, NN+2-opt, và **Google OR-Tools**.

## Cấu trúc
```
config.py                 # toàn bộ tham số (bài toán, PRP, ACO, giao thông, nhiên liệu)
src/
  graph_loader.py         # tải bản đồ thật (OSMnx) + ma trận + hình học đường phố
  problem.py              # Order/Vehicle + sinh instance (gồm đơn động)
  emission_model.py       # mô hình phát thải PRP theo tốc độ + E10 + CO₂
  travel.py               # hệ số tắc đường theo giờ cao điểm
  aco_solver.py           # Hybrid ACO (MMAS + 2-opt/Or-opt/relocate + warm-start)
  baseline.py             # NN, NN+2-opt, OR-Tools
  dynamic_sim.py          # mô phỏng động rolling-horizon
  metrics.py              # chỉ số: quãng đường, nhiên liệu, CO₂, tốc độ, vi phạm TG
  visualize.py            # vẽ lộ trình, hội tụ, đường cong tốc độ–phát thải
experiments/
  run_static.py           # ACO tĩnh + so sánh baseline
  run_dynamic.py          # động vs tĩnh (giá trị của tái tối ưu)
  run_fueltype.py         # so sánh E0/E5/E10
  run_emission.py         # đường cong tốc độ–phát thải + tác động giờ cao điểm
webapp/
  server.py               # máy chủ http.server (không cần Flask)
  index.html              # giao diện Leaflet (bản đồ thật, kiểu Google Maps)
```

## Cài đặt
```bash
pip install -r requirements.txt      # numpy, networkx, matplotlib, osmnx
pip install ortools                  # tùy chọn — bật baseline OR-Tools
```
Lần chạy đầu sẽ tải bản đồ quận Hoàn Kiếm từ OSM và cache vào `data/graph.graphml`.

## Chạy thí nghiệm (in số liệu + lưu ảnh vào `outputs/`)
```bash
python -m experiments.run_static     # Hybrid ACO vs baseline
python -m experiments.run_dynamic    # định tuyến động vs tĩnh
python -m experiments.run_fueltype   # E0 / E5 / E10
python -m experiments.run_emission   # tốc độ–phát thải + giờ cao điểm
```

## Giao diện bản đồ thực (kiểu Google Maps)
```bash
python -m webapp.server              # rồi mở http://localhost:8000
```
Bảng điều khiển cho phép đổi số đơn/xe/tải, loại nhiên liệu (E0/E5/E10), giờ khởi
hành, bật/tắt tắc đường, và chế độ **tĩnh** (lộ trình tối ưu) hoặc **động** (đơn
phát sinh + tái tối ưu). Lộ trình được vẽ **bám theo đường phố thật**; bảng chỉ số
hiển thị nhiên liệu, chi phí, CO₂, tốc độ trung bình.

Hai tính năng quan trọng:
- **Tự chọn điểm trên bản đồ**: chọn "Nguồn điểm → Tự chọn", rồi **nhấp bản đồ**
  để đặt kho (điểm đầu) và các đơn giao. Hệ thống snap mỗi điểm vào nút đường gần
  nhất và tính lộ trình tối ưu.
- **So sánh thuật toán**: chọn Hybrid ACO / OR-Tools / NN+2-opt để đối chiếu trực
  tiếp trên cùng bộ điểm.

Điểm được lấy mặc định **ngẫu nhiên** từ các nút đường OSM (`graph_loader.sample_nodes`);
ở chế độ tự chọn thì lấy từ vị trí người dùng nhấp (`graph_loader.nearest_node`).

## Hạn chế & hướng phát triển
- Hybrid ACO hiện thua OR-Tools (GLS) trên instance tĩnh — cần neighborhood mạnh
  hơn (inter-route 2-opt\*, ALNS) để gom xe tốt hơn.
- Giao thông động dùng hồ sơ giờ cao điểm tổng hợp; có thể thay bằng dữ liệu thật.
- Mở rộng tiềm năng: đa mục tiêu (Pareto chi phí–CO₂), đội xe hỗn hợp EV + E10,
  bất định ngẫu nhiên (stochastic/robust), DRL dẫn dắt heuristic.

## Tham khảo
- Bektaş, Laporte (2011). *The Pollution-Routing Problem*. Transportation Research B.
- Demir, Bektaş, Laporte (2012). *An adaptive large neighborhood search for the PRP*.
- Ichoua, Gendreau, Potvin (2003). *Vehicle dispatching with time-dependent travel times*.
- Solomon (1987). *VRPTW benchmark instances*.
- Dorigo, Stützle (2004). *Ant Colony Optimization* (MAX-MIN Ant System).
