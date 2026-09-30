"""CS F407 Week 2 lab: goal-based warehouse navigation agent.

The agent keeps an explicit goal and a model of the world (the grid). It uses
Breadth-First Search (BFS) to find a shortest collision-free path from S to G.

Run:  python warehouse_agent.py
"""
import sys
import time
from collections import deque

if hasattr(sys.stdout, "reconfigure"):  # absent in some notebook/IDE consoles
    sys.stdout.reconfigure(encoding="utf-8")

WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (d_row, d_col). The order fixes tie-breaking between equal paths.
ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


class WarehouseAgent:
    """Goal-based agent: perceives the map, then plans actions that reach G."""

    def __init__(self, map_text):
        self.grid = [list(row) for row in map_text.strip().splitlines()]
        self._validate()
        self.start = self._find("S")
        self.goal = self._find("G")
        self.expanded = 0  # states expanded by the last search

    def _validate(self):
        if not self.grid or len({len(r) for r in self.grid}) != 1:
            raise ValueError("Map must be non-empty and rectangular.")
        flat = [c for r in self.grid for c in r]
        if flat.count("S") != 1 or flat.count("G") != 1:
            raise ValueError("Map must contain exactly one 'S' and one 'G'.")
        if set(flat) - set("#.SG"):
            raise ValueError("Map may only contain '#', '.', 'S' and 'G'.")

    def _find(self, ch):
        for r, row in enumerate(self.grid):
            if ch in row:
                return (r, row.index(ch))

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) and self.grid[r][c] != "#"

    def successors(self, pos):
        """Legal (action, next_position) pairs from pos."""
        for name, (dr, dc) in ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                yield name, nxt

    def plan(self):
        """BFS. Returns (list of positions, list of actions) or None if unreachable."""
        frontier = deque([self.start])
        parent = {self.start: None}  # position -> (previous position, action)
        self.expanded = 0
        while frontier:
            pos = frontier.popleft()
            self.expanded += 1
            if pos == self.goal:
                path, acts = [pos], []
                while parent[pos] is not None:
                    pos, a = parent[pos]
                    path.append(pos)
                    acts.append(a)
                return path[::-1], acts[::-1]
            for a, nxt in self.successors(pos):
                if nxt not in parent:
                    parent[nxt] = (pos, a)
                    frontier.append(nxt)
        return None

    def render(self, path):
        out = [row[:] for row in self.grid]
        for r, c in path[1:-1]:
            out[r][c] = "*"
        return "\n".join("".join(row) for row in out)


def run(map_text, title):
    print(f"=== {title} ===")
    agent = WarehouseAgent(map_text)
    t0 = time.perf_counter()
    result = agent.plan()
    dt = (time.perf_counter() - t0) * 1000
    if result is None:
        print(f"No path exists from S{agent.start} to G{agent.goal}. "
              f"({agent.expanded} states explored, {dt:.2f} ms)\n")
        return None
    path, acts = result
    print(f"Start {agent.start} -> Goal {agent.goal}")
    print(f"Path length: {len(acts)} moves; states expanded: {agent.expanded}; {dt:.2f} ms")
    print("Path (row, col):", " ".join(map(str, path)))
    print("Actions:", ", ".join(acts))
    print(agent.render(path), "\n")
    return path, acts


def verify(map_text, path, acts):
    """Independent check: every step is one legal move onto a free cell."""
    agent = WarehouseAgent(map_text)
    assert path[0] == agent.start and path[-1] == agent.goal
    for (p, q), a in zip(zip(path, path[1:]), acts):
        assert agent.grid[q[0]][q[1]] != "#"
        assert (q[0] - p[0], q[1] - p[1]) == ACTIONS[a]
    assert len(set(path)) == len(path)


def doubled(map_text):
    """Scale-up experiment: each cell becomes a 2x2 block (S/G kept at one corner)."""
    rows = []
    for line in map_text.splitlines():
        top, bot = "", ""
        for ch in line:
            if ch in "SG":
                top += ch + "."
                bot += ".."
            else:
                top += ch * 2
                bot += ch * 2
        rows += [top, bot]
    return "\n".join(rows)


ALGORITHM_NOTE = """\
Search algorithm: Breadth-First Search (BFS).
Why: every move costs one step, the map is small, fully known and static, so BFS is
complete and always returns a shortest path in O(cells) time. DFS gives no shortest-path
guarantee; Dijkstra/A* add cost handling or heuristics that this problem does not need."""


if __name__ == "__main__":
    print(ALGORITHM_NOTE, "\n")
    res = run(WAREHOUSE, "Given warehouse map")
    verify(WAREHOUSE, *res)
    print("Path verified: legal moves only, no obstacles crossed.\n")

    blocked = [list(r) for r in WAREHOUSE.splitlines()]
    blocked[1][18] = blocked[2][19] = "#"  # wall off both neighbours of G
    blocked = ["".join(r) for r in blocked]
    run("\n".join(blocked), "Test: goal walled off (no path)")

    big = doubled(WAREHOUSE)
    res = run(big, "Test: warehouse twice as large (each cell -> 2x2)")
    verify(big, *res)

    for bad in ["#S#\n#.#", "S..G\n#.."]:
        try:
            WarehouseAgent(bad)
        except ValueError as e:
            print(f"Invalid map rejected: {e}")
