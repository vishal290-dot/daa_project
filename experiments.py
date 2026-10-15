#!/usr/bin/env python3
"""Parts C-E on the disjoint subgraph CSVs produced by preprocess.py.

For each target size: pick SUBS subgraphs whose vertex count is closest to the
target, run PAIRS random source/destination pairs in each, and compare
  Dijkstra | A* (straight-line heuristic) | A* (ALT landmarks)
Reuses the from-scratch algorithms in sp.py (no graph library).

Usage: python3 experiments.py [subgraph_dir] [out_dir]
"""
import csv, math, random, sys, time
from collections import defaultdict
sys.path.insert(0, "/home/claude/work")
import sp

SIZES = [1000, 5000, 10000, 25000, 50000]
SUBS, PAIRS, SEED, K_LM = 3, 5, 7, 8


def read_subgraph_ids(d, size):
    cnt = defaultdict(int)
    with open(f"{d}/vertices_{size}.csv") as f:
        for r in csv.DictReader(f):
            cnt[int(r["subgraph_id"])] += 1
    return cnt


def load_subgraphs(d, size, wanted):
    wanted = set(wanted)
    verts = {p: [] for p in wanted}
    with open(f"{d}/vertices_{size}.csv") as f:
        for r in csv.DictReader(f):
            p = int(r["subgraph_id"])
            if p in wanted:
                verts[p].append((int(r["local_id"]), int(r["node_id"]), float(r["lon"]), float(r["lat"])))
    adjs = {p: [[] for _ in verts[p]] for p in wanted}
    with open(f"{d}/edges_{size}.csv") as f:
        for r in csv.DictReader(f):
            p = int(r["subgraph_id"])
            if p in wanted:
                adjs[p][int(r["local_src"])].append((int(r["local_dst"]), int(r["weight"])))
    return verts, adjs


def planar(verts):
    """lon/lat -> metres (equirectangular around the subgraph centre)."""
    lat0 = sum(v[3] for v in verts) / len(verts)
    lon0 = sum(v[2] for v in verts) / len(verts)
    kx = 111320.0 * math.cos(math.radians(lat0))
    xy = [None] * len(verts)
    for lid, _, lon, lat in verts:
        xy[lid] = ((lon - lon0) * kx, (lat - lat0) * 110574.0)
    return xy


def run(d, out):
    rows, summ = [], []
    for size in SIZES:
        cnt = read_subgraph_ids(d, size)
        chosen = sorted(cnt, key=lambda p: abs(cnt[p] - size))[:SUBS]
        verts, adjs = load_subgraphs(d, size, chosen)
        for p in chosen:
            adj, vs = adjs[p], verts[p]
            xy = planar(vs)
            orig = {lid: nid for lid, nid, _, _ in vs}
            E = sum(len(a) for a in adj)
            tp = time.perf_counter()
            fac_e = sp.euclid_heuristic(xy, adj)      # scale = min(w / length) over this subgraph's edges
            prep_e = time.perf_counter() - tp
            tp = time.perf_counter()
            fac_a = sp.alt_heuristic(adj, K_LM, SEED)
            prep_a = time.perf_counter() - tp
            qr = random.Random(SEED * 1000 + p + size)
            algos = (("Dijkstra", None, 0.0), ("A*-Euclid", fac_e, prep_e), ("A*-ALT8", fac_a, prep_a))
            acc = {a[0]: [] for a in algos}
            for q in range(PAIRS):
                s, t = qr.sample(range(len(adj)), 2)
                ref = None
                for name, fac, _ in algos:
                    t0 = time.perf_counter()
                    c, path, ex, xm = sp.dijkstra(adj, s, t) if fac is None else sp.astar(adj, s, t, fac)
                    ms = (time.perf_counter() - t0) * 1000
                    if ref is None:
                        ref = c
                    assert c == ref, (size, p, s, t, name, c, ref)   # all must be optimal
                    rows.append([size, p, len(adj), q, orig[s], orig[t], name, c, len(path), ex, xm, round(ms, 3)])
                    acc[name].append((c, ex, xm, ms))
            for name, _, prep in algos:
                L = acc[name]
                m = [sum(x[i] for x in L) / len(L) for i in range(4)]
                summ.append([size, p, len(adj), E // 2, name, *m, prep])
            print(f"size {size} subgraph {p} (|V|={len(adj)}) done", flush=True)
    with open(f"{out}/results_disjoint.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_size", "subgraph_id", "V", "query", "src_node", "dst_node", "algo", "cost",
                    "path_vertices", "explored", "examined", "time_ms"])
        w.writerows(rows)
    # aggregate per target size and algorithm (mean over all queries)
    agg = defaultdict(list)
    for r in rows:
        agg[(r[0], r[6])].append(r)
    with open(f"{out}/summary_disjoint.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_size", "algo", "queries", "avg_V", "avg_cost", "avg_explored", "avg_examined",
                    "avg_time_ms", "avg_prep_s"])
        prep = defaultdict(list)
        for s in summ:
            prep[(s[0], s[4])].append(s[9])
        for (size, name), L in agg.items():
            n = len(L)
            w.writerow([size, name, n, sum(r[2] for r in L) / n, sum(r[7] for r in L) / n,
                        sum(r[9] for r in L) / n, sum(r[10] for r in L) / n, sum(r[11] for r in L) / n,
                        sum(prep[(size, name)]) / len(prep[(size, name)])])


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/subgraphs"
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/claude"
    run(d, out)
