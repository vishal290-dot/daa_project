# daa_project
# Shortest-Path Planning on the DIMACS Colorado Road Network (Dijkstra vs A*)

Implements Dijkstra (binary min-heap) and A* (straight-line and landmark heuristics) from scratch, with no
graph libraries, on the 9th DIMACS Challenge road network USA-road-d.COL (435,666 vertices, 521,200 roads),
split into disjoint subgraphs of about 1k, 5k, 10k, 25k and 50k vertices. Includes the dynamic-edge-cost bonus.

## Requirements
Python 3.10+ and `matplotlib` (only for plots: `pip install matplotlib`). No other packages.

## Input data (put in the project folder)
- `USA-road-d.COL.gr.gz` – arcs (`a u v w`)
- `USA-road-d.COL.co` – coordinates (`v id lon*1e6 lat*1e6`)
(Source: http://www.dis.uniroma1.it/~challenge9 . Use your actual file names if they carry a prefix; unzipped `.gr` also works.)

## How to run (in this order, from the project folder)
```
python3 preprocess.py USA-road-d.COL.gr.gz USA-road-d.COL.co -o subgraphs   # 1. partition + CSV export (~30 s)
python3 experiments.py subgraphs .                                          # 2. Parts C-E experiments
python3 dynamic_costs.py subgraphs .                                        # 3. bonus: w = alpha*d + beta*c
python3 plot_graphs.py subgraphs . .                                        # 4. graph maps + performance plot
python3 plot_comparison.py . .                                              # 5. Dijkstra vs A* comparison plots
python3 dynamic.py subgraphs results_dynamic.csv                            # (optional) stale-heuristic supplement
```

## Files
| File | Purpose |
|---|---|
| `sp.py` | Library: graph loading, Dijkstra, A*, straight-line and landmark (ALT) heuristics. Imported by the other scripts; not run directly. |
| `preprocess.py` | Partitions the graph into disjoint connected subgraphs; writes `vertices_N.csv`, `edges_N.csv` per size and `stats.csv`. |
| `experiments.py` | Runs Dijkstra and both A* versions on the subgraphs (3 subgraphs x 5 random pairs per size). |
| `dynamic_costs.py` | Bonus: dynamic edge costs, original vs new route comparison. |
| `dynamic.py` | Optional supplement: stale vs refreshed heuristics when costs change. |
| `plot_graphs.py`, `plot_comparison.py` | Figures. |
| `REPORT.pdf` / `.docx` / `.md` | Full report: Parts A-F, bonus, appendix. |

## Output files
- `subgraphs/vertices_N.csv` (node_id, subgraph_id, local_id, lon, lat, out_degree), `subgraphs/edges_N.csv` (subgraph_id, src, dst, local_src, local_dst, weight, length_m), `subgraphs/stats.csv`
- `results_disjoint.csv` (every query), `summary_disjoint.csv` (averages per size and algorithm)
- `results_dynamic_costs.csv`, `route_change_example.png/.txt`; `results_dynamic.csv` (optional)
- `performance_disjoint.png`, `graph_map_N.png`, `plot_edges_vs_size.png`, `plot_vertices_vs_size.png`, `plot_time_vs_size.png`, `plot_time_vs_size_loglog.png`, `plot_comparison_all.png`

## Method in brief
- **Graph:** vertex = intersection (with lon/lat), edge = road segment, weight = segment length (about 10 units per metre); duplicate arcs collapsed to the minimum; roads stored in both directions.
- **Partitioning:** BFS region growing, small leftovers merged into neighbours; every subgraph is connected and sizes are approximate (about 0.5x-1.5x the target). Arcs between subgraphs are dropped (counted in `stats.csv` as `cut_arcs`).
- **Dijkstra:** heapq with lazy deletion and early exit; O((V+E) log V).
- **A\*:** (1) straight-line, h = c * distance with c = minimum weight/length ratio (admissible and consistent); (2) ALT, 8 landmarks, h = max |d(L,D) - d(L,v)| (admissible and consistent).
- **Metrics:** shortest-path cost, path, vertices explored (settled), edges examined (relaxations attempted), execution time. All algorithms are asserted to return the same cost on every query.

## Notes
- Timings are single-run Python and averages over few random queries, so they are indicative; the exact numbers will differ slightly on your machine.
- `experiments.py` chooses the subgraphs closest to each target size, so sizes in the results are about 1,002, 5,048, 10,191, 25,624 and 52,105 vertices.
