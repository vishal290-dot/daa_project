"""Shortest-path planning on the DIMACS USA-road-d.COL (Colorado) road network.

Everything algorithmic (graph build, Dijkstra, A*, landmark heuristic) is
implemented from scratch with only the standard library.  No graph libraries.
"""
import gzip, heapq, math, random, time, csv, sys, os
from collections import deque

INF = float("inf")


# ----------------------------------------------------------------- Part B
def load_graph(path):
    """Read DIMACS .gr(.gz) -> adjacency list adj[u] = [(v, w), ...] (0-indexed).
    Parallel arcs are collapsed to the minimum weight. Returns (adj, n_records)."""
    op = gzip.open if path.endswith(".gz") else open
    adj, n, records = None, 0, 0
    best = {}
    with op(path, "rt") as f:
        for line in f:
            c = line[0]
            if c == "p":
                n = int(line.split()[2])
                adj = [[] for _ in range(n)]
            elif c == "a":
                _, u, v, w = line.split()
                u, v, w = int(u) - 1, int(v) - 1, int(w)
                records += 1
                k = (u, v)
                if k in best:
                    if w < best[k]:
                        best[k] = w
                else:
                    best[k] = w
    for (u, v), w in best.items():
        adj[u].append((v, w))
    return adj, records


def load_coords(path):
    """Optional DIMACS .co file -> list of (x, y) (degrees * 1e6)."""
    op = gzip.open if path.endswith(".gz") else open
    xy = None
    with op(path, "rt") as f:
        for line in f:
            if line[0] == "p":
                xy = [None] * int(line.split()[-1])
            elif line[0] == "v":
                _, i, x, y = line.split()
                xy[int(i) - 1] = (int(x), int(y))
    return xy


def stats(adj):
    n = len(adj)
    deg = [len(a) for a in adj]
    m = sum(deg)
    return {"V": n, "arcs": m, "undirected_edges": m // 2,
            "avg_degree": m / n, "max_degree": max(deg)}


def bfs_subgraph(adj, center, size):
    """Connected induced subgraph: first `size` vertices reached by BFS from
    `center`, relabelled 0..size-1. Returns (sub_adj, old_ids)."""
    seen = {center: 0}
    order = [center]
    dq = deque([center])
    while dq and len(order) < size:
        u = dq.popleft()
        for v, _ in adj[u]:
            if v not in seen:
                seen[v] = len(order)
                order.append(v)
                dq.append(v)
                if len(order) >= size:
                    break
    sub = [[] for _ in order]
    for old, new in seen.items():
        sub[new] = [(seen[v], w) for v, w in adj[old] if v in seen]
    return sub, order


# ----------------------------------------------------------------- Part C
def dijkstra(adj, s, t):
    """Binary-heap Dijkstra with lazy deletion, early exit at t.
    Returns (cost, path, explored, examined)."""
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    done = [False] * n
    dist[s] = 0
    pq = [(0, s)]
    explored = examined = 0
    pop, push = heapq.heappop, heapq.heappush
    while pq:
        d, u = pop(pq)
        if done[u]:
            continue
        done[u] = True
        explored += 1
        if u == t:
            break
        for v, w in adj[u]:
            examined += 1
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                push(pq, (nd, v))
    return dist[t], _path(parent, s, t, dist), explored, examined


def _path(parent, s, t, dist):
    if dist[t] == INF:
        return []
    p = [t]
    while p[-1] != s:
        p.append(parent[p[-1]])
    return p[::-1]


def sssp(adj, s):
    """Full single-source Dijkstra (used only for landmark preprocessing)."""
    dist = [INF] * len(adj)
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ----------------------------------------------------------------- Part D
def astar(adj, s, t, h):
    """A* with heuristic factory h(t) -> function v -> estimate.
    Heuristic is consistent, so each vertex is settled once.
    Returns (cost, path, explored, examined)."""
    hv = h(t)
    n = len(adj)
    g = [INF] * n
    parent = [-1] * n
    done = [False] * n
    g[s] = 0
    pq = [(hv(s), s)]
    explored = examined = 0
    pop, push = heapq.heappop, heapq.heappush
    while pq:
        _, u = pop(pq)
        if done[u]:
            continue
        done[u] = True
        explored += 1
        if u == t:
            break
        gu = g[u]
        for v, w in adj[u]:
            examined += 1
            ng = gu + w
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                push(pq, (ng + hv(v), v))
    return g[t], _path(parent, s, t, g), explored, examined


def euclid_heuristic(xy, adj):
    """Straight-line heuristic, scaled by the smallest weight/length ratio over
    all edges so that it never overestimates (admissible + consistent)."""
    ratio = min(w / d for u in range(len(adj)) for v, w in adj[u]
                if (d := math.dist(xy[u], xy[v])) > 0)
    def factory(t):
        tx, ty = xy[t]
        return lambda v: ratio * math.hypot(xy[v][0] - tx, xy[v][1] - ty)
    return factory


def alt_heuristic(adj, k=8, seed=0):
    """ALT (A*, Landmarks, Triangle inequality).  Choose k landmarks by
    farthest-point selection, precompute exact distances from each.
    On a symmetric graph, h(v) = max_L |d(L,t) - d(L,v)| <= d(v,t) (triangle
    inequality) -> admissible; it is also consistent."""
    rnd = random.Random(seed)
    n = len(adj)
    first = rnd.randrange(n)
    d0 = sssp(adj, first)
    cur = max(range(n), key=lambda i: d0[i] if d0[i] < INF else -1)
    tables, mind = [], [INF] * n
    for _ in range(k):
        dl = sssp(adj, cur)
        tables.append(dl)
        mind = [min(a, b) for a, b in zip(mind, dl)]
        cur = max(range(n), key=lambda i: mind[i])
    def factory(t):
        dt = [T[t] for T in tables]
        def hv(v):
            best = 0
            for T, x in zip(tables, dt):
                d = x - T[v]
                if d < 0:
                    d = -d
                if d > best:
                    best = d
            return best
        return hv
    return factory


# ----------------------------------------------------------------- Part E
def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t0) * 1000


def run(gr_path, co_path, out_csv, sizes, n_pairs=10, k_landmarks=8, seed=42):
    t0 = time.perf_counter()
    adj, records = load_graph(gr_path)
    print(f"loaded {len(adj)} V, {records} arcs in {time.perf_counter()-t0:.1f}s", flush=True)
    full = stats(adj)
    print("FULL", full, flush=True)
    xy = load_coords(co_path) if co_path and os.path.exists(co_path) else None
    rnd = random.Random(seed)
    center = rnd.randrange(len(adj))
    rows, summary = [], []
    for size in sizes:
        if size >= len(adj):
            sub, order = adj, list(range(len(adj)))
        else:
            sub, order = bfs_subgraph(adj, center, size)
        st = stats(sub)
        hname = "euclid" if xy else f"ALT-{k_landmarks}"
        tp = time.perf_counter()
        if xy:
            sxy = [xy[o] for o in order]
            fac = euclid_heuristic(sxy, sub)
        else:
            fac = alt_heuristic(sub, k_landmarks, seed)
        prep = time.perf_counter() - tp
        print(f"size {len(sub)}: {st}  heuristic prep {prep:.2f}s", flush=True)
        qr = random.Random(seed + size)
        agg = {"Dijkstra": [], "A*": []}
        for q in range(n_pairs):
            s, t = qr.sample(range(len(sub)), 2)
            (cd, pd, ed, xd), td = timed(dijkstra, sub, s, t)
            (ca, pa, ea, xa), ta = timed(astar, sub, s, t, fac)
            assert cd == ca, (size, s, t, cd, ca)       # A* must be optimal
            for name, c, p, e, x, tm in (("Dijkstra", cd, pd, ed, xd, td),
                                         ("A*", ca, pa, ea, xa, ta)):
                rows.append([len(sub), q, order[s], order[t], name, c, len(p), e, x, round(tm, 3)])
                agg[name].append((c, e, x, tm))
        for name, lst in agg.items():
            m = [sum(v[i] for v in lst) / len(lst) for i in range(4)]
            summary.append([len(sub), st["undirected_edges"], st["avg_degree"], st["max_degree"],
                            name, *m, prep if name == "A*" else 0.0])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V", "query", "src_orig_id", "dst_orig_id", "algo", "cost", "path_vertices",
                    "explored", "examined", "time_ms"])
        w.writerows(rows)
    with open(out_csv.replace(".csv", "_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V", "undirected_edges", "avg_degree", "max_degree", "algo", "avg_cost",
                    "avg_explored", "avg_examined", "avg_time_ms", "heuristic_prep_s"])
        w.writerows(summary)
    return full, summary


if __name__ == "__main__":
    gr = sys.argv[1]
    co = sys.argv[2] if len(sys.argv) > 2 else None
    run(gr, co, "results.csv", [1000, 5000, 10000, 25000, 50000, 100000, 435666])
