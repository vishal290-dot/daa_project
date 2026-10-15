#!/usr/bin/env python3
"""Bonus task: dynamic edge costs   w(u,v) = alpha*d(u,v) + beta*c(u,v).

d(u,v) = original road length (dataset distance units).
c(u,v) = dynamic delay on the road, expressed in the same units: c = rho * d, where rho >= 0 is
         a congestion factor (rho = 0 -> free flow, rho = 2 -> the road takes 3x as long at beta=1).
Scenarios (which roads get c > 0):
  random    : 20% of roads, rho ~ U(1, 3)                (diffuse congestion)
  incident  : every road inside a disc around the MIDDLE of the original route (radius = 15% of the
              S-D straight-line distance, >= 300 m), rho ~ U(2, 4)   (localized jam / accident / weather)
Parameter sweep: alpha = 1, beta in {0, 0.5, 1, 2, 4}  (beta = 0 reproduces the original network).

For each query: original shortest route (beta = 0) vs new optimal route under the new weights.
Metrics: route changed?, cost of keeping the old route under new weights vs new optimal cost,
physical length (sum of d) of both routes, detour %, shared-edge overlap %, and algorithm effort.
Since alpha*d + beta*c >= d for alpha = 1, beta >= 0, A* heuristics built on d stay admissible.

Usage: python3 dynamic_costs.py [subgraph_dir] [out_dir]
"""
import csv, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp, experiments as ex

SIZES, PAIRS, SEED = [10000, 25000], 20, 5
ALPHA, BETAS = 1.0, [0.0, 0.5, 1.0, 2.0, 4.0]
SCENARIOS = ["random", "incident"]


def congestion(adj, xy, scen, s, t, path, rnd):
    """Return dict {(u,v): rho} for BOTH directions of each affected road."""
    rho = {}
    if scen == "random":
        for u in range(len(adj)):
            for v, _ in adj[u]:
                if u < v and rnd.random() < 0.20:
                    r = rnd.uniform(1, 3)
                    rho[(u, v)] = rho[(v, u)] = r
    else:
        mid = xy[path[len(path) // 2]]
        R = max(0.15 * math.dist(xy[s], xy[t]), 300.0)
        for u in range(len(adj)):
            if math.dist(xy[u], mid) <= R:
                for v, _ in adj[u]:
                    if u < v and math.dist(xy[v], mid) <= R:
                        r = rnd.uniform(2, 4)
                        rho[(u, v)] = rho[(v, u)] = r
    return rho


def reweight(adj, rho, alpha, beta):
    return [[(v, max(1, round(alpha * w + beta * w * rho.get((u, v), 0.0)))) for v, w in a]
            for u, a in enumerate(adj)]


def path_cost(adj, path):
    tot = 0
    for a, b in zip(path, path[1:]):
        tot += next(w for v, w in adj[a] if v == b)
    return tot


def edge_set(path):
    return {frozenset(e) for e in zip(path, path[1:])}


def pick_query(adj, qr, min_hops=25):
    for _ in range(200):
        s, t = qr.sample(range(len(adj)), 2)
        c, p, _, _ = sp.dijkstra(adj, s, t)
        if len(p) >= min_hops:
            return s, t, p
    return s, t, p


def main(d, out):
    rows, example = [], None
    for size in SIZES:
        cnt = ex.read_subgraph_ids(d, size)
        pid = sorted(cnt, key=lambda q: abs(cnt[q] - size))[0]
        verts, adjs = ex.load_subgraphs(d, size, [pid])
        adj, vs = adjs[pid], verts[pid]
        xy = ex.planar(vs)
        orig_id = {lid: nid for lid, nid, _, _ in vs}
        fac_e = sp.euclid_heuristic(xy, adj)        # built on d only
        fac_a = sp.alt_heuristic(adj, 8, SEED)
        qr = random.Random(SEED + size)
        queries = [pick_query(adj, qr) for _ in range(PAIRS)]
        for scen in SCENARIOS:
            for beta in BETAS:
                acc = dict(changed=0, keep=0.0, new=0.0, d_old=0.0, d_new=0.0, overlap=0.0,
                           hops_old=0, hops_new=0, ex_d=0, ex_e=0, ex_a=0, t_d=0.0, t_e=0.0, t_a=0.0)
                for qi, (s, t, p0) in enumerate(queries):
                    rho = congestion(adj, xy, scen, s, t, p0, random.Random(SEED * 31 + qi + size)) if beta > 0 else {}
                    new = reweight(adj, rho, ALPHA, beta)
                    t0 = time.perf_counter(); c1, p1, e1, _ = sp.dijkstra(new, s, t); td = (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter(); c2, _, e2, _ = sp.astar(new, s, t, fac_e); te = (time.perf_counter() - t0) * 1000
                    t0 = time.perf_counter(); c3, _, e3, _ = sp.astar(new, s, t, fac_a); ta = (time.perf_counter() - t0) * 1000
                    assert c1 == c2 == c3, (scen, beta, qi, c1, c2, c3)
                    keep = path_cost(new, p0)
                    acc["changed"] += p1 != p0
                    acc["keep"] += keep; acc["new"] += c1
                    acc["d_old"] += path_cost(adj, p0); acc["d_new"] += path_cost(adj, p1)
                    a, b = edge_set(p0), edge_set(p1)
                    acc["overlap"] += 100 * len(a & b) / len(a)
                    acc["hops_old"] += len(p0); acc["hops_new"] += len(p1)
                    acc["ex_d"] += e1; acc["ex_e"] += e2; acc["ex_a"] += e3
                    acc["t_d"] += td; acc["t_e"] += te; acc["t_a"] += ta
                    if size == 10000 and scen == "incident" and beta == 2.0 and example is None and p1 != p0:
                        example = dict(xy=xy, s=s, t=t, p0=p0, p1=p1, rho=rho, keep=keep, c1=c1,
                                       d0=path_cost(adj, p0), d1=path_cost(adj, p1), orig=orig_id)
                n = PAIRS
                rows.append([size, len(adj), scen, ALPHA, beta, n, 100 * acc["changed"] / n,
                             acc["keep"] / n, acc["new"] / n, 100 * (acc["keep"] - acc["new"]) / acc["keep"],
                             acc["d_old"] / n, acc["d_new"] / n, 100 * (acc["d_new"] - acc["d_old"]) / acc["d_old"],
                             acc["overlap"] / n, acc["hops_old"] / n, acc["hops_new"] / n,
                             acc["ex_d"] / n, acc["ex_e"] / n, acc["ex_a"] / n,
                             acc["t_d"] / n, acc["t_e"] / n, acc["t_a"] / n])
                print(f"size {size} {scen} beta={beta} done", flush=True)
    hdr = ["target_size", "V", "scenario", "alpha", "beta", "queries", "route_changed_pct",
           "old_route_cost_under_new_w", "new_optimal_cost", "saving_from_rerouting_pct",
           "old_route_distance", "new_route_distance", "distance_detour_pct", "edge_overlap_pct",
           "old_hops", "new_hops", "explored_dijkstra", "explored_astar_euclid", "explored_astar_alt",
           "ms_dijkstra", "ms_astar_euclid", "ms_astar_alt"]
    with open(f"{out}/results_dynamic_costs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    if example:
        plot(example, out)


def plot(e, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xy = e["xy"]
    fig, ax = plt.subplots(figsize=(7, 7))
    pts = set(e["p0"]) | set(e["p1"])
    zone = {u for (u, v) in e["rho"]} | {v for (u, v) in e["rho"]}
    for (u, v) in e["rho"]:
        if u < v:
            ax.plot([xy[u][0], xy[v][0]], [xy[u][1], xy[v][1]], color="#f5b7b1", lw=0.7, zorder=1)
    ax.plot(*zip(*[xy[u] for u in e["p0"]]), color="#2471a3", lw=2.5, label="original route", zorder=3)
    ax.plot(*zip(*[xy[u] for u in e["p1"]]), color="#c0392b", lw=2.5, ls="--", label="new route (beta=2)", zorder=4)
    ax.scatter(*xy[e["s"]], c="green", s=70, zorder=5, label="S"); ax.scatter(*xy[e["t"]], c="black", s=70, zorder=5, label="D")
    ax.set_aspect("equal"); ax.set_title("Incident zone (pink roads) forces a detour"); ax.legend(); ax.set_xlabel("metres"); ax.set_ylabel("metres")
    plt.tight_layout(); plt.savefig(f"{out}/route_change_example.png", dpi=130)
    with open(f"{out}/route_change_example.txt", "w") as f:
        o = e["orig"]
        f.write(f"S={o[e['s']]} D={o[e['t']]} (original node ids), alpha=1, beta=2, incident scenario\n")
        f.write(f"old route: {len(e['p0'])} vertices, distance {e['d0']}, cost under new weights {e['keep']}\n")
        f.write(f"new route: {len(e['p1'])} vertices, distance {e['d1']}, cost under new weights {e['c1']}\n")
        f.write("old path: " + str([o[u] for u in e["p0"]]) + "\nnew path: " + str([o[u] for u in e["p1"]]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/subgraphs",
         sys.argv[2] if len(sys.argv) > 2 else "/home/claude")
