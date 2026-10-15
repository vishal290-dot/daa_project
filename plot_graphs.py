#!/usr/bin/env python3
"""Make the pictures.
  1. performance_disjoint.png : Dijkstra vs A* (from summary_disjoint.csv)
  2. graph_map_<N>.png        : road map of one subgraph per size, with a Dijkstra shortest route
Usage: python3 plot_graphs.py [subgraph_dir] [csv_dir] [out_dir]
"""
import csv, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sp, experiments as ex

d = sys.argv[1] if len(sys.argv) > 1 else "subgraphs"
cd = sys.argv[2] if len(sys.argv) > 2 else "."
out = sys.argv[3] if len(sys.argv) > 3 else "."

# 1. performance plot
R = list(csv.DictReader(open(f"{cd}/summary_disjoint.csv")))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for key, lab, a in (("avg_explored", "Avg vertices explored", ax[0]), ("avg_examined", "Avg edges examined", ax[1]),
                    ("avg_time_ms", "Avg time (ms)", ax[2])):
    for algo, c in (("Dijkstra", "#c0392b"), ("A*-Euclid", "#e69f00"), ("A*-ALT8", "#2471a3")):
        r = [x for x in R if x["algo"] == algo]
        a.plot([int(x["target_size"]) for x in r], [float(x[key]) for x in r], "o-", color=c, label=algo)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Subgraph size |V|"); a.set_title(lab); a.grid(alpha=.3)
ax[0].legend(); plt.tight_layout(); plt.savefig(f"{out}/performance_disjoint.png", dpi=130); plt.close()

# 2. graph maps
for size in (1000, 5000, 10000, 25000, 50000):
    cnt = ex.read_subgraph_ids(d, size)
    pid = sorted(cnt, key=lambda q: abs(cnt[q] - size))[0]
    verts, adjs = ex.load_subgraphs(d, size, [pid])
    adj, vs = adjs[pid], verts[pid]
    pos = {lid: (lon, lat) for lid, _, lon, lat in vs}
    segs = [(pos[u], pos[v]) for u in range(len(adj)) for v, _ in adj[u] if u < v]
    qr = random.Random(size)
    while True:
        s, t = qr.sample(range(len(adj)), 2)
        cost, path, ex_n, _ = sp.dijkstra(adj, s, t)
        if len(path) >= 15: break
    fig, a = plt.subplots(figsize=(7, 7))
    a.add_collection(LineCollection(segs, colors="#9aa5b1", linewidths=0.4))
    a.plot(*zip(*[pos[u] for u in path]), color="#c0392b", lw=2.2, label=f"Dijkstra shortest path ({len(path)} vertices)")
    a.scatter(*pos[s], c="green", s=60, zorder=5, label="S"); a.scatter(*pos[t], c="black", s=60, zorder=5, label="D")
    a.autoscale(); a.set_aspect("equal"); a.set_xlabel("longitude"); a.set_ylabel("latitude")
    a.set_title(f"Subgraph with {len(adj):,} vertices, {len(segs):,} roads"); a.legend(loc="best")
    plt.tight_layout(); plt.savefig(f"{out}/graph_map_{size}.png", dpi=130); plt.close()
    print("wrote graph_map_%d.png" % size)
