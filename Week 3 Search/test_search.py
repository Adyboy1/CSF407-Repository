"""CS F407 Week 3 lab: tests for the LLM-generated program astar_llm.py.

astar_llm.py is used unmodified. This file holds the test maps, an independent
path checker and an independent shortest-path oracle (so the LLM's BFS is not
the only thing A* is compared with).

Run:  python test_search.py
"""
import random
import sys
from collections import deque

import astar_llm as L

if hasattr(sys.stdout, "reconfigure"):  # absent in some notebook/IDE consoles
    sys.stdout.reconfigure(encoding="utf-8")

TRIVIAL = """\
#####
#SG##
#####"""

NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

ALTERNATIVES = """\
#######
#S...G#
#.###.#
#.....#
#######"""

HEURISTICS = {
    "Manhattan": L.h_manhattan,
    "Zero (h=0)": L.h_zero,
    "Euclidean": L.h_euclidean,
    "2 x Manhattan": L.h_double_manhattan,
}
MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


def astar(p, name="Manhattan"):
    return L.astar(p, HEURISTICS[name])


def oracle_length(p):
    """Independent shortest path length (own BFS on the raw text); None if unreachable."""
    grid, dist, q = p.grid, {p.start: 0}, deque([p.start])
    while q:
        r, c = q.popleft()
        if (r, c) == p.goal:
            return dist[(r, c)]
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if (0 <= n[0] < len(grid) and 0 <= n[1] < len(grid[0])
                    and grid[n[0]][n[1]] != "#" and n not in dist):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return None


def valid(p, res):
    """Replay the actions from S: every move legal, ends on G, length matches."""
    if not res["found"]:
        return res["states"] == [] and res["length"] is None
    pos = p.start
    if res["states"][0] != pos or len(res["states"]) != len(res["actions"]) + 1:
        return False
    for a, nxt in zip(res["actions"], res["states"][1:]):
        dr, dc = MOVES[a]
        if (pos[0] + dr, pos[1] + dc) != nxt or p.grid[nxt[0]][nxt[1]] == "#":
            return False
        pos = nxt
    return pos == p.goal and res["length"] == len(res["actions"])


def summary(res):
    if not res["found"]:
        return f"found=False  expanded={res['expanded']}"
    return f"found=True  length={res['length']}  expanded={res['expanded']}"


def random_grid(rng, rows=9, cols=13, p_wall=0.28):
    g = [["#" if rng.random() < p_wall else "." for _ in range(cols)]
         for _ in range(rows)]
    g[0][0], g[-1][-1] = "S", "G"
    return "\n".join("".join(r) for r in g)


def main():
    wh = L.Warehouse(L.MAP)
    ra, rb = astar(wh), L.bfs(wh)

    print("=== Test 1: original warehouse (A*, Manhattan) ===")
    print(summary(ra), "| valid:", valid(wh, ra),
          "| oracle length:", oracle_length(wh))
    print("Path:", " ".join(f"({r},{c})" for r, c in ra["states"]))
    print("Actions:", ", ".join(ra["actions"]))
    print(L.draw(wh, ra["states"]))

    for title, text, expect in [
        ("Test 2: trivial case", TRIVIAL, "found, length 1"),
        ("Test 3: no solution", NO_SOLUTION, "not found, terminates"),
        ("Test 4: alternative paths", ALTERNATIVES, "found, shortest = 4"),
    ]:
        p = L.Warehouse(text)
        a, b = astar(p), L.bfs(p)
        print(f"\n=== {title}  (expected: {expect}) ===")
        print(text)
        print("A*: ", summary(a), "| valid:", valid(p, a))
        print("BFS:", summary(b), "| valid:", valid(p, b))
        print("Independent oracle length:", oracle_length(p),
              "| A* matches oracle:", a["length"] == oracle_length(p))

    print("\n=== Task 5: BFS vs A* on the warehouse ===")
    print(f"{'Measure':<16}{'BFS':>8}{'A*':>8}")
    print(f"{'Solution found':<16}{str(rb['found']):>8}{str(ra['found']):>8}")
    print(f"{'Path length':<16}{rb['length']:>8}{ra['length']:>8}")
    print(f"{'States expanded':<16}{rb['expanded']:>8}{ra['expanded']:>8}")

    print("\n=== Task 6: heuristic investigation (warehouse) ===")
    print(f"{'Heuristic':<16}{'Found':>7}{'Length':>8}{'Expanded':>10}")
    for name in HEURISTICS:
        r = astar(wh, name)
        print(f"{name:<16}{str(r['found']):>7}{r['length']:>8}{r['expanded']:>10}"
              f"   valid={valid(wh, r)}")

    print("\n=== Validation: 500 random 9x13 grids vs independent oracle ===")
    rng = random.Random(407)
    n_sol = n_uns = 0
    bad = {n: 0 for n in HEURISTICS}
    bad["BFS"] = 0
    for _ in range(500):
        p = L.Warehouse(random_grid(rng))
        best = oracle_length(p)
        if best is None:
            n_uns += 1
        else:
            n_sol += 1
        for name in HEURISTICS:
            r = astar(p, name)
            if r["length"] != best or not valid(p, r):
                bad[name] += 1
        b = L.bfs(p)
        if b["length"] != best or not valid(p, b):
            bad["BFS"] += 1
    print(f"solvable={n_sol} unsolvable={n_uns}")
    print("Grids where the result differs from the oracle (length/validity):")
    for name, n in bad.items():
        print(f"  {name:<16}{n}")

    print("\n=== Open grids (9x13, 15% walls): mean over solvable grids ===")
    rng = random.Random(407)
    tot = {n: [0, 0] for n in list(HEURISTICS) + ["BFS"]}
    worse, solvable = None, 0
    for _ in range(500):
        text = random_grid(rng, p_wall=0.15)
        p = L.Warehouse(text)
        best = oracle_length(p)
        if best is None:
            continue
        solvable += 1
        for name in HEURISTICS:
            r = astar(p, name)
            tot[name][0] += r["expanded"]
            tot[name][1] += r["length"]
            if name == "2 x Manhattan" and r["length"] > best and worse is None:
                worse = (text, best, r["length"])
        b = L.bfs(p)
        tot["BFS"][0] += b["expanded"]
        tot["BFS"][1] += b["length"]
    print(f"solvable grids: {solvable}")
    print(f"{'Algorithm/h':<16}{'Mean expanded':>15}{'Mean length':>13}")
    for name, (e, l) in tot.items():
        print(f"{name:<16}{e / solvable:>15.1f}{l / solvable:>13.2f}")
    if worse:
        print(f"\nExample where 2 x Manhattan is NOT shortest "
              f"(shortest {worse[1]}, got {worse[2]}):")
        print(worse[0])


if __name__ == "__main__":
    main()
