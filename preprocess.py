#!/usr/bin/env python3
"""Partition a DIMACS road graph (.gr / .gr.gz) into DISJOINT connected subgraphs
of approximately a target number of vertices, and export CSV files.

Input format (9th DIMACS Challenge):
    p sp <n> <m>          header
    a <u> <v> <w>         directed arc u -> v with positive weight w (1-indexed)

    v <id> <x> <y>        (.co file) longitude, latitude in degrees * 1e6

Usage:
    python3 preprocess.py USA-road-d.COL.gr.gz USA-road-d.COL.co [-o out_dir] [-s 1000 5000 10000 25000 50000]

Output (out_dir/):
    vertices_<N>.csv   node_id, subgraph_id, local_id, lon, lat, out_degree
    edges_<N>.csv      subgraph_id, src, dst, local_src, local_dst, weight, length_m
                       (length_m = great-circle distance between the endpoints)
    stats.csv          one row for the whole graph, then per size: one "ALL" row
                       (aggregate) plus one row per subgraph

Method (per target size N):
  1. Region growing: BFS from a seed over still-unassigned vertices until N
     vertices are taken. The next seed is taken from the previous region's
     frontier so regions tile the map contiguously.
  2. Any region smaller than MIN_FRAC*N (leftover pockets) is merged into its
     smallest adjacent region, so every subgraph is connected and sizes lie
     roughly in [MIN_FRAC*N, (1+MIN_FRAC)*N). Isolated components of the original
     graph (if any) stay separate.
  3. Only arcs whose two endpoints are in the same subgraph are exported
     (arcs crossing between subgraphs are counted in stats as cut_arcs).
Node ids in the CSVs are the ORIGINAL 1-indexed DIMACS ids; local_id is
0..size-1 within each subgraph.
"""
import argparse, csv, gzip, math, os, sys
from collections import deque

MIN_FRAC = 0.5


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    n, best = 0, {}
    with op(path, "rt") as f:
        for line in f:
            if line[0] == "p":
                n = int(line.split()[2])
            elif line[0] == "a":
                _, u, v, w = line.split()
                u, v, w = int(u) - 1, int(v) - 1, int(w)
                if u == v:
                    continue                      # drop self loops
                k = (u, v)
                if k not in best or w < best[k]:  # collapse parallel arcs -> min
                    best[k] = w
    adj = [[] for _ in range(n)]
    for (u, v), w in best.items():
        adj[u].append((v, w))
    return adj


def load_coords(path, n):
    """.co(.gz) -> (lon, lat) lists in degrees."""
    op = gzip.open if path.endswith(".gz") else open
    lon, lat = [0.0] * n, [0.0] * n
    with op(path, "rt") as f:
        for line in f:
            if line[0] == "v":
                _, i, x, y = line.split()
                lon[int(i) - 1], lat[int(i) - 1] = int(x) / 1e6, int(y) / 1e6
    return lon, lat


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def partition(adj, size):
    n = len(adj)
    part = [-1] * n
    pools = []                      # pools[pid] = list of vertices
    cand = deque()
    scan = 0
    while True:
        seed = -1
        while cand:
            c = cand.popleft()
            if part[c] == -1:
                seed = c
                break
        if seed == -1:
            while scan < n and part[scan] != -1:
                scan += 1
            if scan == n:
                break
            seed = scan
        pid = len(pools)
        members, dq = [seed], deque([seed])
        part[seed] = pid
        while dq and len(members) < size:
            u = dq.popleft()
            for v, _ in adj[u]:
                if part[v] == -1:
                    part[v] = pid
                    members.append(v)
                    dq.append(v)
                    if len(members) >= size:
                        break
        for u in dq:                # leftover frontier -> future seeds
            for v, _ in adj[u]:
                if part[v] == -1:
                    cand.append(v)
        pools.append(members)

    # merge small leftover regions into their smallest neighbouring region
    parent = list(range(len(pools)))
    sz = [len(p) for p in pools]
    find = lambda x: x if parent[x] == x else find(parent[x])
    for pid in sorted(range(len(pools)), key=lambda p: sz[p]):
        r = find(pid)
        if sz[r] >= MIN_FRAC * size:
            continue
        nb = {}
        for u in pools[pid]:
            for v, _ in adj[u]:
                rv = find(part[v])
                if rv != r:
                    nb[rv] = sz[rv]
        if not nb:
            continue                # isolated component: keep as is
        t = min(nb, key=nb.get)
        parent[r] = t
        sz[t] += sz[r]
    # relabel final ids 0..k-1 (ordered by first vertex id)
    remap, final = {}, [0] * n
    for u in range(n):
        r = find(part[u])
        if r not in remap:
            remap[r] = len(remap)
        final[u] = remap[r]
    return final, len(remap)


def components(adj, members, part, pid):
    seen, c = set(), 0
    for s in members:
        if s in seen:
            continue
        c += 1
        seen.add(s)
        st = [s]
        while st:
            u = st.pop()
            for v, _ in adj[u]:
                if part[v] == pid and v not in seen:
                    seen.add(v)
                    st.append(v)
    return c


STAT_COLS = ["scope", "target_size", "subgraph_id", "vertices", "arcs", "undirected_edges",
             "avg_degree", "max_degree", "connected_components", "cut_arcs",
             "min_weight", "avg_weight", "max_weight", "lon_min", "lon_max", "lat_min", "lat_max",
             "min_weight_per_m", "avg_weight_per_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("coords", help="DIMACS .co file with node coordinates")
    ap.add_argument("-o", "--out", default="subgraphs")
    ap.add_argument("-s", "--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 25000, 50000])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    adj = load(a.graph)
    n = len(adj)
    lon, lat = load_coords(a.coords, n)
    def ratio_stats(pairs):   # pairs of (w, length_m); min ratio = admissible heuristic scale
        r = [w / d for w, d in pairs if d > 0]
        return (min(r), sum(r) / len(r)) if r else ("", "")
    allw = [w for u in adj for _, w in u]
    rows = []
    deg = [len(x) for x in adj]
    fr = ratio_stats((w, haversine(lon[u], lat[u], lon[v], lat[v])) for u in range(n) for v, w in adj[u])
    rows.append(["full_graph", "", "", n, len(allw), len(allw) // 2, len(allw) / n, max(deg), "",
                 0, min(allw), sum(allw) / len(allw), max(allw), min(lon), max(lon), min(lat), max(lat), *fr])
    print(f"loaded {n} vertices, {len(allw)} arcs")

    for size in a.sizes:
        part, k = partition(adj, size)
        members = [[] for _ in range(k)]
        for u, p in enumerate(part):
            members[p].append(u)
        local = [0] * n
        for p in range(k):
            for i, u in enumerate(members[p]):
                local[u] = i

        with open(f"{a.out}/vertices_{size}.csv", "w", newline="") as fv, \
             open(f"{a.out}/edges_{size}.csv", "w", newline="") as fe:
            wv, we = csv.writer(fv), csv.writer(fe)
            wv.writerow(["node_id", "subgraph_id", "local_id", "lon", "lat", "out_degree"])
            we.writerow(["subgraph_id", "src", "dst", "local_src", "local_dst", "weight", "length_m"])
            agg = [0, 0, 0, 0, 0, 0]  # arcs, cut, wsum, wmin, wmax, comps
            agg[3] = float("inf")
            agg_r = [float("inf"), 0.0, 0]   # min ratio, sum ratio, count
            per = []
            for p in range(k):
                arcs = cut = wsum = 0
                wmin, wmax, maxdeg = float("inf"), 0, 0
                inner = {u: 0 for u in members[p]}
                rmin, rsum, rcnt = float("inf"), 0.0, 0
                for u in members[p]:
                    for v, w in adj[u]:
                        if part[v] == p:
                            d = haversine(lon[u], lat[u], lon[v], lat[v])
                            we.writerow([p, u + 1, v + 1, local[u], local[v], w, round(d, 3)])
                            if d > 0:
                                r = w / d
                                rmin, rsum, rcnt = min(rmin, r), rsum + r, rcnt + 1
                            arcs += 1
                            inner[u] += 1
                            wsum += w
                            wmin, wmax = min(wmin, w), max(wmax, w)
                        else:
                            cut += 1
                for u in members[p]:
                    wv.writerow([u + 1, p, local[u], lon[u], lat[u], inner[u]])
                    maxdeg = max(maxdeg, inner[u])
                comps = components(adj, members[p], part, p)
                nv = len(members[p])
                per.append([f"subgraph", size, p, nv, arcs, arcs // 2, arcs / nv, maxdeg, comps,
                            cut, wmin if arcs else "", wsum / arcs if arcs else "", wmax if arcs else "",
                            min(lon[u] for u in members[p]), max(lon[u] for u in members[p]),
                            min(lat[u] for u in members[p]), max(lat[u] for u in members[p]),
                            rmin if rcnt else "", rsum / rcnt if rcnt else ""])
                agg_r = [min(agg_r[0], rmin), agg_r[1] + rsum, agg_r[2] + rcnt]
                agg[0] += arcs; agg[1] += cut; agg[2] += wsum
                if arcs:
                    agg[3] = min(agg[3], wmin); agg[4] = max(agg[4], wmax)
                agg[5] += comps
        vs = [r[3] for r in per]
        rows.append([f"summary(k={k};mean_size={sum(vs)/k:.0f};min={min(vs)};max={max(vs)})", size, "ALL",
                     n, agg[0], agg[0] // 2, agg[0] / n, max(r[7] for r in per), agg[5], agg[1],
                     agg[3], agg[2] / agg[0], agg[4], min(lon), max(lon), min(lat), max(lat),
                     agg_r[0], agg_r[1] / agg_r[2]])
        rows.extend(per)
        print(f"size {size}: {k} subgraphs, vertices min/mean/max = {min(vs)}/{sum(vs)/k:.0f}/{max(vs)}, "
              f"all connected = {all(r[8] == 1 for r in per)}, cut arcs = {agg[1]}")

    with open(f"{a.out}/stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(STAT_COLS)
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
