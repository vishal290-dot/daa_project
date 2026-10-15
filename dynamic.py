#!/usr/bin/env python3
"""Bonus: dynamic edge costs (traffic / congestion).

Edge weights change after the heuristics were precomputed. For each scenario we compare,
on the same queries, Dijkstra against A* with
  * STALE heuristics  (Euclidean scale / ALT tables computed on the ORIGINAL weights)
  * REFRESHED heuristics (recomputed on the new weights; refresh time reported)
and count how often a heuristic yields a non-optimal path.

Theory: if costs only INCREASE, d_new >= d_old, so stale h <= d_old <= d_new stays
admissible and consistent. If any cost DECREASES, stale h can exceed the true cost -> A* may
return a sub-optimal path (and the closed-set shortcut breaks).

Usage: python3 dynamic.py [subgraph_dir] [out_csv]
"""
import csv, random, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp, experiments as ex

SEED, PAIRS = 11, 20
SIZES = [10000, 25000]
# name, share of undirected roads changed, (low, high) multiplier range
SCENARIOS = [("increase-light", 0.20, (1.5, 4.0)),
             ("increase-heavy", 0.50, (2.0, 6.0)),
             ("mixed-with-decreases", 0.30, (0.3, 3.0))]


def perturb(adj, share, lohi, rnd):
    """Symmetric multiplier on a random share of undirected roads; integer weights >= 1."""
    new = [list(a) for a in adj]
    idx = [{v: i for i, (v, _) in enumerate(a)} for a in adj]
    for u in range(len(adj)):
        for v, w in adj[u]:
            if u < v and rnd.random() < share:
                m = rnd.uniform(*lohi)
                for a, b in ((u, v), (v, u)):
                    i = idx[a].get(b)
                    if i is not None:
                        new[a][i] = (b, max(1, round(adj[a][i][1] * m)))
    return new


def main(d, out):
    rows = []
    for size in SIZES:
        cnt = ex.read_subgraph_ids(d, size)
        p = sorted(cnt, key=lambda q: abs(cnt[q] - size))[0]
        verts, adjs = ex.load_subgraphs(d, size, [p])
        adj, vs = adjs[p], verts[p]
        xy = ex.planar(vs)
        fac_e0 = sp.euclid_heuristic(xy, adj)           # stale = built on original weights
        fac_a0 = sp.alt_heuristic(adj, 8, SEED)
        for name, share, lohi in SCENARIOS:
            rnd = random.Random(SEED + size)
            new = perturb(adj, share, lohi, rnd)
            t0 = time.perf_counter(); fac_e1 = sp.euclid_heuristic(xy, new); pe = time.perf_counter() - t0
            t0 = time.perf_counter(); fac_a1 = sp.alt_heuristic(new, 8, SEED); pa = time.perf_counter() - t0
            variants = (("Dijkstra", None, 0), ("A*-Euclid stale", fac_e0, 0), ("A*-Euclid refreshed", fac_e1, pe),
                        ("A*-ALT8 stale", fac_a0, 0), ("A*-ALT8 refreshed", fac_a1, pa))
            acc = {v[0]: [0, 0, 0.0, 0, 0.0] for v in variants}   # explored, examined, ms, suboptimal, excess%
            qr = random.Random(SEED * 7 + size)
            for _ in range(PAIRS):
                s, t = qr.sample(range(len(new)), 2)
                ref = None
                for vname, fac, _ in variants:
                    t0 = time.perf_counter()
                    c, _, e, x = sp.dijkstra(new, s, t) if fac is None else sp.astar(new, s, t, fac)
                    ms = (time.perf_counter() - t0) * 1000
                    ref = c if ref is None else ref
                    a = acc[vname]
                    a[0] += e; a[1] += x; a[2] += ms
                    if c > ref:
                        a[3] += 1; a[4] += 100 * (c - ref) / ref
            for vname, fac, prep in variants:
                a = acc[vname]
                rows.append([size, len(new), name, vname, PAIRS, a[0] / PAIRS, a[1] / PAIRS, a[2] / PAIRS,
                             a[3], a[4] / max(1, a[3]), prep])
            print(f"size {size} {name} done", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_size", "V", "scenario", "algorithm", "queries", "avg_explored", "avg_examined",
                    "avg_time_ms", "suboptimal_paths", "avg_excess_cost_pct_when_suboptimal", "refresh_time_s"])
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/subgraphs",
         sys.argv[2] if len(sys.argv) > 2 else "/home/claude/results_dynamic.csv")
