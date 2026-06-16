# Emission-Aware Dynamic Routing & Smart Consolidation (E10)

An urban delivery problem with **dynamic order arrivals, congestion and rush
hours**, optimizing **fuel & CO₂** in the context of **Hanoi switching to E10
gasoline (2026)**.

Algorithmic core: **Hybrid Ant Colony Optimization** (MAX-MIN Ant System + local
search) on top of a **Time-Dependent VRPTW**, with the objective being a
**speed-dependent emission model (Pollution-Routing Problem)**. Map data is
**real**, from OpenStreetMap (OSMnx).

## Academic highlights
- **Speed-based PRP emission model** (Bektaş & Laporte 2011; CMEM): fuel depends
  on speed–load–acceleration–grade, with an **optimal speed** v\* ≈ 44 km/h.
- **Time-dependent traffic**: congestion factor by rush hour → lower speed →
  higher fuel (measured **+16.6%** on the same route during peak hours).
- **Fuel types E0/E5/E10**: E10 uses ~3.6% more volume but is **~3.8% cheaper**
  and **cuts ~6.7% fossil CO₂**.
- **Dynamic routing**: rolling-horizon, dynamic orders, pheromone warm-start.
- **Benchmark**: compared against Nearest-Neighbor, NN+2-opt, and **Google OR-Tools**.

## Structure
```
config.py                 # all parameters (problem, PRP, ACO, traffic, fuel)
src/
  graph_loader.py         # load real map (OSMnx) + matrices + street geometry
  problem.py              # Order/Vehicle + instance generation (incl. dynamic orders)
  emission_model.py       # speed-based PRP emission model + E10 + CO₂
  travel.py               # rush-hour congestion factor
  aco_solver.py           # Hybrid ACO (MMAS + 2-opt/Or-opt/relocate + warm-start)
  baseline.py             # NN, NN+2-opt, OR-Tools
  dynamic_sim.py          # rolling-horizon dynamic simulation
  metrics.py              # metrics: distance, fuel, CO₂, speed, TW violations
  visualize.py            # plot routes, convergence, speed–emission curve
  benchmark.py            # Solomon loader + DistanceModel (distance objective)
experiments/
  run_static.py           # static ACO + baseline comparison
  run_dynamic.py          # dynamic vs static (value of re-optimization)
  run_fueltype.py         # E0/E5/E10 comparison
  run_emission.py         # speed–emission curve + rush-hour impact
  run_benchmark.py        # standard Solomon VRPTW benchmark (6 classes)
  run_statistics.py       # multi-seed: mean ± CI95 + Wilcoxon signed-rank
data/solomon/             # 6 real Solomon instances (C101/C201/R101/R201/RC101/RC201)
webapp/
  server.py               # http.server backend (no Flask required)
  index.html              # Leaflet UI (real map, Google-Maps style)
```

## Installation
```bash
pip install -r requirements.txt      # numpy, networkx, matplotlib, osmnx
pip install ortools                  # optional — enables the OR-Tools baseline
```
The first run downloads the Hoan Kiem district map from OSM and caches it to
`data/graph.graphml`.

## Running experiments (prints metrics + saves images to `outputs/`)
```bash
python -m experiments.run_static     # Hybrid ACO vs baselines
python -m experiments.run_dynamic    # dynamic vs static routing
python -m experiments.run_fueltype   # E0 / E5 / E10
python -m experiments.run_emission   # speed–emission + rush hour
python -m experiments.run_benchmark 25   # Solomon (size 25 | 50 | 100)
python -m experiments.run_statistics 10  # multi-seed + CI95 + Wilcoxon
```

### Statistical reliability (added)
**Multi-seed N=30 (Hanoi, 25 orders), mean ± CI95, PRP fuel objective:**

| Algorithm | Fuel (L) | Distance (km) | Vehicles |
|---|---|---|---|
| **Hybrid ACO** | 2.154 [2.083, 2.225] | 23.45 | 3.8 |
| NN+2-opt | 2.718 [2.609, 2.826] | 29.46 | 4.3 |
| OR-Tools | 2.011 [1.949, 2.073] | 21.68 | 3.6 |

**Wilcoxon signed-rank (paired)**: ACO beats NN+2-opt **+20.7%, wins 30/30,
p≈1.9e-9**; trails OR-Tools by **7.1%, p≈9.3e-9**. The CIs of ACO and NN+2-opt
**do not overlap**. Chart: `outputs/statistics_fuel.png`.

**Standard Solomon benchmark (100 customers), distance objective:**
- **Tightly clustered** class (C101): ACO **854 vs best-known 829 (+3%), exactly 10
  vehicles** — near-optimal.
- Across feasible instances: ACO is ~6% from BKS (note: the solver uses a **fixed
  fleet** and minimizes distance, unlike Solomon's hierarchical "minimize vehicles
  first" objective, so the vehicle count may differ from BKS).
- **Limitation at the 100-customer scale**: on tight-TW classes (R1/RC1) ACO still
  **violates time windows** (marked `*`) — it needs a stronger feasibility-repair
  mechanism (ALNS). At 25/50 customers ACO is consistently feasible.

## Real-map interface (Google-Maps style)
```bash
python -m webapp.server              # then open http://localhost:8000
```
<img width="1692" height="893" alt="image" src="https://github.com/user-attachments/assets/47d2073c-b17a-422e-8536-e6c402d2f78a" />

The control panel lets you change the number of orders/vehicles/capacity, fuel
type (E0/E5/E10), departure hour, toggle congestion, and switch between **static**
(optimal route) and **dynamic** (dynamic orders + re-optimization) modes. Routes
are drawn **following the real streets**; the metrics panel shows fuel, cost, CO₂
and average speed.

Two key features:
- **Pick points on the map**: choose "Point source → Manual", then **click the map**
  to place the depot (first point) and the delivery orders. The system snaps each
  point to the nearest road node and computes the optimal route.
- **Compare algorithms**: select Hybrid ACO / OR-Tools / NN+2-opt to compare them
  directly on the same set of points.
<img width="1692" height="858" alt="image" src="https://github.com/user-attachments/assets/62fcf93f-1b4c-41e9-89f7-951b85e94efc" />

By default, points are sampled **randomly** from OSM road nodes
(`graph_loader.sample_nodes`); in manual mode they come from where the user clicks
(`graph_loader.nearest_node`).

## Results & figures

The figures below are generated into the `outputs/` folder when running the experiments.

**Speed – emission curve (PRP)** — there is an optimal speed v\* ≈ 44 km/h; driving
too slow (congestion) or too fast both waste fuel:

![Speed–emission curve](outputs/speed_emission_curve.png)

**Optimal routes (Hybrid ACO, Hoan Kiem map)** and the **ACO convergence curve:**

![Optimal routes](outputs/static_routes.png)
![ACO convergence](outputs/static_convergence.png)

**Executed routes in dynamic mode** (dynamic orders + re-optimization):

![Dynamic routes](outputs/dynamic_executed.png)

**Algorithm comparison over 30 instances (± 95% confidence interval)** — ACO beats
NN+2-opt significantly and trails OR-Tools by ~7%:

![Multi-seed statistics](outputs/statistics_fuel.png)

**Standard Solomon VRPTW benchmark (total distance, lower = better):**

![Solomon benchmark](outputs/benchmark_solomon_50.png)

## Limitations & future work
- Hybrid ACO still trails OR-Tools by ~5.9% (narrowed from ~38%); ALNS is needed to
  close the gap, especially on R/RC classes with tight time windows.
- Dynamic traffic uses a synthetic rush-hour profile; it could be replaced with real data.
- Potential extensions: multi-objective (cost–CO₂ Pareto), mixed EV + E10 fleet,
  stochastic/robust uncertainty, DRL-guided heuristics.

## References
- Bektaş, Laporte (2011). *The Pollution-Routing Problem*. Transportation Research B.
- Demir, Bektaş, Laporte (2012). *An adaptive large neighborhood search for the PRP*.
- Ichoua, Gendreau, Potvin (2003). *Vehicle dispatching with time-dependent travel times*.
- Solomon (1987). *VRPTW benchmark instances*.
- Dorigo, Stützle (2004). *Ant Colony Optimization* (MAX-MIN Ant System).
