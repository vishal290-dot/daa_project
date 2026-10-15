#!/usr/bin/env python3
"""Dijkstra vs A* comparison plots from summary_disjoint.csv.
Writes: plot_edges_vs_size.png, plot_vertices_vs_size.png, plot_time_vs_size.png,
        plot_time_vs_size_loglog.png, plot_comparison_all.png
Usage: python3 plot_comparison.py [csv_dir] [out_dir]
"""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cd = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else "."
R = list(csv.DictReader(open(f"{cd}/summary_disjoint.csv")))
STY = (("Dijkstra", "#c0392b", "o"), ("A*-Euclid", "#e69f00", "s"), ("A*-ALT8", "#2471a3", "^"))
LAB = {"Dijkstra": "Dijkstra", "A*-Euclid": "A* (straight-line)", "A*-ALT8": "A* (landmarks)"}
PLOTS = [("avg_examined", "Edges examined vs graph size", "Avg edges examined", False, "plot_edges_vs_size.png"),
         ("avg_explored", "Vertices explored vs graph size", "Avg vertices explored", False, "plot_vertices_vs_size.png"),
         ("avg_time_ms", "Execution time vs graph size", "Avg execution time (ms)", False, "plot_time_vs_size.png"),
         ("avg_time_ms", "Execution time vs graph size (log-log)", "Avg execution time (ms)", True, "plot_time_vs_size_loglog.png")]

def draw(a, key, title, ylab, log):
    for algo, c, m in STY:
        r = [x for x in R if x["algo"] == algo]
        xs, ys = [float(x["avg_V"]) for x in r], [float(x[key]) for x in r]
        a.plot(xs, ys, marker=m, color=c, lw=2, label=LAB[algo])
        if not log:
            for x, y in zip(xs, ys):
                pass
    if log:
        a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("Graph size (number of vertices)"); a.set_ylabel(ylab); a.set_title(title)
    a.grid(alpha=.3, which="both"); a.legend()

for key, title, ylab, log, fn in PLOTS:
    fig, a = plt.subplots(figsize=(6.5, 4.5)); draw(a, key, title, ylab, log)
    plt.tight_layout(); plt.savefig(f"{out}/{fn}", dpi=140); plt.close()
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for a, (key, title, ylab, log, fn) in zip(axs.flat, PLOTS):
    draw(a, key, title, ylab, log)
plt.tight_layout(); plt.savefig(f"{out}/plot_comparison_all.png", dpi=140)
print("done")
