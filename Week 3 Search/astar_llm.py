import heapq
import math
from collections import deque

MAP = """#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

MOVES = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]


class Warehouse:
    def __init__(self, text):
        self.grid = [list(line) for line in text.strip().splitlines()]
        starts, goals = [], []
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == 'S':
                    starts.append((r, c))
                elif ch == 'G':
                    goals.append((r, c))
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Map must contain exactly one S and one G")
        self.start = starts[0]
        self.goal = goals[0]

    def is_goal(self, state):
        return state == self.goal

    def successors(self, state):
        r, c = state
        for action, dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[nr]) \
                    and self.grid[nr][nc] != '#':
                yield action, (nr, nc), 1


# Heuristics: h(state, goal)
def h_zero(s, g):
    return 0


def h_manhattan(s, g):
    return abs(s[0] - g[0]) + abs(s[1] - g[1])


def h_euclidean(s, g):
    return math.hypot(s[0] - g[0], s[1] - g[1])


def h_double_manhattan(s, g):
    return 2 * h_manhattan(s, g)


def reconstruct(parent, goal):
    states, actions = [goal], []
    cur = goal
    while parent[cur] is not None:
        prev, action = parent[cur]
        states.append(prev)
        actions.append(action)
        cur = prev
    states.reverse()
    actions.reverse()
    return states, actions


def make_result(found, states, actions, expanded):
    return {
        "found": found,
        "states": states,
        "actions": actions,
        "length": len(actions) if found else None,
        "expanded": expanded,
    }


def astar(problem, heuristic):
    start, goal = problem.start, problem.goal
    counter = 0
    frontier = [(heuristic(start, goal), counter, start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        _, _, state = heapq.heappop(frontier)
        if state in closed:
            continue
        if problem.is_goal(state):
            states, actions = reconstruct(parent, state)
            return make_result(True, states, actions, expanded)
        closed.add(state)
        expanded += 1
        for action, nxt, cost in problem.successors(state):
            new_g = g[state] + cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = (state, action)
                counter += 1
                heapq.heappush(frontier,
                               (new_g + heuristic(nxt, goal), counter, nxt))
    return make_result(False, [], [], expanded)


def bfs(problem):
    start = problem.start
    frontier = deque([start])
    parent = {start: None}
    in_frontier_or_seen = {start}
    closed = set()
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state in closed:
            continue
        if problem.is_goal(state):
            states, actions = reconstruct(parent, state)
            return make_result(True, states, actions, expanded)
        closed.add(state)
        expanded += 1
        for action, nxt, cost in problem.successors(state):
            if nxt not in in_frontier_or_seen:
                in_frontier_or_seen.add(nxt)
                parent[nxt] = (state, action)
                frontier.append(nxt)
    return make_result(False, [], [], expanded)


def draw(problem, states):
    grid = [row[:] for row in problem.grid]
    for r, c in states:
        if grid[r][c] == '.':
            grid[r][c] = '*'
    return "\n".join("".join(row) for row in grid)


def report(name, problem, res):
    print(f"== {name} ==")
    print("Solution found:", res["found"])
    if res["found"]:
        print("Path length:", res["length"])
        print("States:", res["states"])
        print("Actions:", res["actions"])
    print("States expanded:", res["expanded"])
    if res["found"]:
        print(draw(problem, res["states"]))


def main():
    problem = Warehouse(MAP)
    report("A* (Manhattan)", problem, astar(problem, h_manhattan))


if __name__ == "__main__":
    main()
