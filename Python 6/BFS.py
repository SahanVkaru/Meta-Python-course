# -*- coding: utf-8 -*-
# =============================================================================
#  ICT4133 - Artificial Intelligence | Practical 06
#  Search Strategies and Propositional Logic
# =============================================================================

import heapq
import time
from collections import deque
from itertools import product

# =============================================================================
#  PART A - UNINFORMED SEARCH ALGORITHMS
# =============================================================================

# -----------------------------------------------------------------------------
#  1. Breadth-First Search (BFS) - Basic Traversal
# -----------------------------------------------------------------------------
def bfs(graph, start):
    """BFS traversal - visits nodes level by level."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


# -----------------------------------------------------------------------------
#  Task 7: BFS - Return Shortest Path Between Two Nodes
# -----------------------------------------------------------------------------
def bfs_shortest_path(graph, start, goal):
    """Returns the shortest path from start to goal using BFS."""
    visited = {start}
    queue = deque([[start]])          # each element is a full path
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None                       # no path found


# -----------------------------------------------------------------------------
#  Tasks 8 & 9: BFS on a 2D Grid Maze
#  0 = free cell, 1 = blocked
# -----------------------------------------------------------------------------
def bfs_grid(grid, start, goal):
    """BFS on a 2D grid; returns (path, path_length) or (None, -1)."""
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    queue = deque([[start]])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up/down/left/right

    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == goal:
            return path, len(path) - 1
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited \
                    and grid[nr][nc] == 0:
                visited.add((nr, nc))
                queue.append(path + [(nr, nc)])
    return None, -1


# -----------------------------------------------------------------------------
#  2. Depth-First Search (DFS) - Recursive
# -----------------------------------------------------------------------------
def dfs(graph, start, visited=None):
    """Recursive DFS traversal."""
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            order.extend(dfs(graph, neighbor, visited))
    return order


# Iterative DFS
def dfs_iterative(graph, start):
    """Iterative DFS using an explicit stack."""
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            stack.extend(reversed(graph.get(node, [])))
    return order


# -----------------------------------------------------------------------------
#  Task 10: Cycle Detection Using DFS
# -----------------------------------------------------------------------------
def has_cycle(graph):
    """Returns True if graph (directed) contains a cycle, False otherwise."""
    visited = set()
    rec_stack = set()           # nodes in the current recursion stack

    def dfs_cycle(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.discard(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs_cycle(node):
                return True
    return False


# -----------------------------------------------------------------------------
#  Task 11: Topological Sort Using DFS (for DAGs)
# -----------------------------------------------------------------------------
def topological_sort(graph):
    """Returns a topological ordering of a DAG using DFS."""
    visited = set()
    stack = []

    def dfs_topo(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_topo(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs_topo(node)
    return stack[::-1]


# -----------------------------------------------------------------------------
#  Uniform Cost Search (UCS)
# -----------------------------------------------------------------------------
def ucs(graph, start, goal):
    """UCS - expands node with lowest cumulative cost first.
    graph: {node: [(neighbor, cost), ...]}
    Returns (path, total_cost, nodes_expanded).
    """
    heap = [(0, start, [start])]    # (cost, node, path)
    visited = {}
    nodes_expanded = 0
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        nodes_expanded += 1
        if node == goal:
            return path, cost, nodes_expanded
        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or visited[neighbor] > new_cost:
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
    return None, float('inf'), nodes_expanded


# -----------------------------------------------------------------------------
#  Iterative Deepening DFS (IDDFS)
# -----------------------------------------------------------------------------
def iddfs(graph, start, goal, max_depth=50):
    """Iterative deepening DFS; returns path or None."""
    def dls(node, goal, depth, path, visited):
        if node == goal:
            return path
        if depth == 0:
            return None
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                result = dls(neighbor, goal, depth - 1,
                             path + [neighbor], visited)
                if result is not None:
                    return result
                visited.discard(neighbor)
        return None

    for depth in range(max_depth + 1):
        visited = {start}
        result = dls(start, goal, depth, [start], visited)
        if result is not None:
            return result
    return None


# =============================================================================
#  PART B - INFORMED SEARCH ALGORITHMS
# =============================================================================

# -----------------------------------------------------------------------------
#  3. Greedy Best-First Search
# -----------------------------------------------------------------------------
def greedy_best_first(graph, start, goal, heuristic):
    """Greedy Best-First Search using heuristic h(n).
    graph: {node: [neighbor, ...]} (unweighted).
    Returns (path, nodes_expanded).
    """
    visited = set()
    heap = [(heuristic[start], start, [start])]
    nodes_expanded = 0
    while heap:
        _, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        if node == goal:
            return path, nodes_expanded
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(heap,
                    (heuristic[neighbor], neighbor, path + [neighbor]))
    return None, nodes_expanded


# -----------------------------------------------------------------------------
#  4. A* Search
# -----------------------------------------------------------------------------
def a_star(graph, start, goal, heuristic):
    """A* Search - f(n) = g(n) + h(n).
    graph: {node: [(neighbor, cost), ...]}
    Returns (path, total_cost, nodes_expanded).
    """
    heap = [(heuristic[start], 0, start, [start])]   # (f, g, node, path)
    g_cost = {start: 0}
    nodes_expanded = 0
    while heap:
        f, g, node, path = heapq.heappop(heap)
        if g > g_cost.get(node, float('inf')):
            continue
        nodes_expanded += 1
        if node == goal:
            return path, g, nodes_expanded
        for neighbor, cost in graph.get(node, []):
            new_g = g + cost
            if new_g < g_cost.get(neighbor, float('inf')):
                g_cost[neighbor] = new_g
                f_cost = new_g + heuristic[neighbor]
                heapq.heappush(heap,
                    (f_cost, new_g, neighbor, path + [neighbor]))
    return None, float('inf'), nodes_expanded


# -----------------------------------------------------------------------------
#  Task 3 (extended): A* for 8-Puzzle
# -----------------------------------------------------------------------------
def manhattan_distance_puzzle(state, goal):
    """Manhattan distance heuristic for 8-puzzle."""
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                for gi in range(3):
                    for gj in range(3):
                        if goal[gi][gj] == val:
                            distance += abs(i - gi) + abs(j - gj)
    return distance

def get_next_states_puzzle(state):
    """Generate all valid next states of an 8-puzzle."""
    state = [list(row) for row in state]
    blank_r = blank_c = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                blank_r, blank_c = i, j
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    states = []
    for dr, dc in moves:
        nr, nc = blank_r + dr, blank_c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = [row[:] for row in state]
            new_state[blank_r][blank_c], new_state[nr][nc] = \
                new_state[nr][nc], new_state[blank_r][blank_c]
            states.append(tuple(tuple(r) for r in new_state))
    return states

def a_star_puzzle(start, goal):
    """A* for 8-puzzle. Returns (path_of_states, moves, nodes_expanded)."""
    start = tuple(tuple(r) for r in start)
    goal_t = tuple(tuple(r) for r in goal)

    heap = [(manhattan_distance_puzzle(start, goal), 0, start, [start])]
    g_cost = {start: 0}
    nodes_expanded = 0

    while heap:
        _, g, state, path = heapq.heappop(heap)
        if g > g_cost.get(state, float('inf')):
            continue
        nodes_expanded += 1
        if state == goal_t:
            return path, g, nodes_expanded
        for next_state in get_next_states_puzzle(state):
            new_g = g + 1
            if new_g < g_cost.get(next_state, float('inf')):
                g_cost[next_state] = new_g
                h = manhattan_distance_puzzle(next_state, goal)
                heapq.heappush(heap, (new_g + h, new_g, next_state,
                                      path + [next_state]))
    return None, -1, nodes_expanded


# -----------------------------------------------------------------------------
#  Task 12: Generate All Possible Next States of 8-Puzzle (no revisits)
# -----------------------------------------------------------------------------
def generate_next_states(state, visited_states=None):
    """Returns all unvisited next states from current 8-puzzle state."""
    if visited_states is None:
        visited_states = set()
    next_states = get_next_states_puzzle(state)
    return [s for s in next_states if s not in visited_states]


# -----------------------------------------------------------------------------
#  Task 13: Run BFS / DFS / UCS / A* on Same Graph - Compare
# -----------------------------------------------------------------------------
def compare_search_algorithms(graph_unweighted, graph_weighted,
                               heuristic, start, goal):
    """Runs BFS, DFS, UCS, A* on the same graph and prints comparison."""
    separator = "-" * 55

    # --- BFS ---
    t0 = time.perf_counter()
    visited_bfs = {start}
    queue = deque([[start]])
    bfs_path, bfs_cost, bfs_expanded = None, 0, 0
    while queue:
        path = queue.popleft()
        node = path[-1]
        bfs_expanded += 1
        if node == goal:
            bfs_path = path
            break
        for nb in graph_unweighted.get(node, []):
            if nb not in visited_bfs:
                visited_bfs.add(nb)
                queue.append(path + [nb])
    bfs_cost = len(bfs_path) - 1 if bfs_path else -1
    t_bfs = time.perf_counter() - t0

    # --- DFS ---
    t0 = time.perf_counter()
    dfs_path_result = [None]
    dfs_expanded = [0]
    def _dfs(node, goal, path, visited):
        dfs_expanded[0] += 1
        if node == goal:
            dfs_path_result[0] = path[:]
            return True
        for nb in graph_unweighted.get(node, []):
            if nb not in visited:
                visited.add(nb)
                path.append(nb)
                if _dfs(nb, goal, path, visited):
                    return True
                path.pop()
        return False
    _dfs(start, goal, [start], {start})
    dfs_cost = len(dfs_path_result[0]) - 1 if dfs_path_result[0] else -1
    t_dfs = time.perf_counter() - t0

    # --- UCS ---
    t0 = time.perf_counter()
    ucs_path, ucs_cost, ucs_expanded = ucs(graph_weighted, start, goal)
    t_ucs = time.perf_counter() - t0

    # --- A* ---
    t0 = time.perf_counter()
    astar_path, astar_cost, astar_expanded = a_star(graph_weighted, start,
                                                     goal, heuristic)
    t_astar = time.perf_counter() - t0

    print(separator)
    print(f"{'Algorithm':<10} {'Path':<25} {'Cost':<6} {'Expanded':<10} {'Time(s)'}")
    print(separator)
    def fmt(p): return str(p) if p else "None"
    print(f"{'BFS':<10} {fmt(bfs_path):<25} {bfs_cost:<6} {bfs_expanded:<10} {t_bfs:.6f}")
    print(f"{'DFS':<10} {fmt(dfs_path_result[0]):<25} {dfs_cost:<6} {dfs_expanded[0]:<10} {t_dfs:.6f}")
    print(f"{'UCS':<10} {fmt(ucs_path):<25} {ucs_cost:<6} {ucs_expanded:<10} {t_ucs:.6f}")
    print(f"{'A*':<10} {fmt(astar_path):<25} {astar_cost:<6} {astar_expanded:<10} {t_astar:.6f}")
    print(separator)


# =============================================================================
#  PART B (continued) - Task 14: Game Tree DFS - All Root-to-Leaf Paths
# =============================================================================
def print_game_tree_paths(tree, root):
    """DFS traversal of a game tree; prints all root-to-leaf paths."""
    paths = []
    def dfs_paths(node, path):
        path = path + [node]
        children = tree.get(node, [])
        if not children:            # leaf node
            paths.append(path)
            return
        for child in children:
            dfs_paths(child, path)
    dfs_paths(root, [])
    print("All root-to-leaf paths:")
    for p in paths:
        print(" ->".join(str(n) for n in p))
    return paths


# =============================================================================
#  Task 15: A* on 2D Grid with Manhattan Distance - Visual Output
# =============================================================================
def a_star_grid(grid, start, goal):
    """A* on a 2D grid using Manhattan distance. Prints visual path."""
    rows, cols = len(grid), len(grid[0])

    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    heap = [(h(start), 0, start, [start])]
    g_cost = {start: 0}
    directions = [(-1,0),(1,0),(0,-1),(0,1)]

    while heap:
        f, g, pos, path = heapq.heappop(heap)
        if g > g_cost.get(pos, float('inf')):
            continue
        if pos == goal:
            # Print visual grid
            visual = [row[:] for row in grid]
            for r, c in path:
                visual[r][c] = '*'
            sr, sc = start
            gr, gc = goal
            visual[sr][sc] = 'S'
            visual[gr][gc] = 'G'
            print("A* Grid Path (S=Start, G=Goal, *=Path, 1=Obstacle, 0=Free):")
            for row in visual:
                print(' '.join(str(cell) for cell in row))
            print(f"Path: {path}")
            print(f"Path length: {len(path) - 1} steps")
            return path
        r, c = pos
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
                new_g = g + 1
                neighbor = (nr, nc)
                if new_g < g_cost.get(neighbor, float('inf')):
                    g_cost[neighbor] = new_g
                    heapq.heappush(heap,
                        (new_g + h(neighbor), new_g, neighbor,
                         path + [neighbor]))
    print("No path found.")
    return None


# =============================================================================
#  PART C - PROPOSITIONAL LOGIC
# =============================================================================

# --- Basic Connectives ---
def negation(p):       return not p
def conjunction(p, q): return p and q
def disjunction(p, q): return p or q
def xor(p, q):         return (p and not q) or (not p and q)   # manual XOR
def implication(p, q): return (not p) or q
def biconditional(p, q): return p == q


# -----------------------------------------------------------------------------
#  Task 16: Truth Table for (p /\ q) -> r
# -----------------------------------------------------------------------------
def truth_table_task16():
    print("\nTask 16: Truth Table for (p /\ q) -> r")
    print(f"{'p':<8}{'q':<8}{'r':<8}{'(p/\q)->r'}")
    print("-" * 35)
    for p, q, r in product([True, False], repeat=3):
        result = implication(conjunction(p, q), r)
        print(f"{str(p):<8}{str(q):<8}{str(r):<8}{result}")


# -----------------------------------------------------------------------------
#  Task 17: Truth Table for (p \/ q) /\ (~r \/ p)
# -----------------------------------------------------------------------------
def truth_table_task17():
    print("\nTask 17: Truth Table for (p \/ q) /\ (~r \/ p)")
    print(f"{'p':<8}{'q':<8}{'r':<8}{'Result'}")
    print("-" * 35)
    for p, q, r in product([True, False], repeat=3):
        result = conjunction(disjunction(p, q), disjunction(negation(r), p))
        print(f"{str(p):<8}{str(q):<8}{str(r):<8}{result}")


# -----------------------------------------------------------------------------
#  Task 18: Truth Table for (p XOR q) <-> r  (manual XOR)
# -----------------------------------------------------------------------------
def truth_table_task18():
    print("\nTask 18: Truth Table for (p XOR q) <-> r  (manual XOR)")
    print(f"{'p':<8}{'q':<8}{'r':<8}{'(pXORq)<->r'}")
    print("-" * 35)
    for p, q, r in product([True, False], repeat=3):
        result = biconditional(xor(p, q), r)
        print(f"{str(p):<8}{str(q):<8}{str(r):<8}{result}")


# -----------------------------------------------------------------------------
#  Task 19: Truth Table for (p /\ ~q) \/ (r /\ ~p)
# -----------------------------------------------------------------------------
def truth_table_task19():
    print("\nTask 19: Truth Table for (p /\ ~q) \/ (r /\ ~p)")
    print(f"{'p':<8}{'q':<8}{'r':<8}{'Result'}")
    print("-" * 35)
    for p, q, r in product([True, False], repeat=3):
        result = disjunction(
            conjunction(p, negation(q)),
            conjunction(r, negation(p))
        )
        print(f"{str(p):<8}{str(q):<8}{str(r):<8}{result}")


# -----------------------------------------------------------------------------
#  Task 4 (Lab Task 04.1): Truth Table for (p \/ ~q) /\ ~p
# -----------------------------------------------------------------------------
def truth_table_task04():
    print("\nTask 04: Truth Table for (p \/ ~q) /\ ~p")
    print(f"{'p':<8}{'q':<8}{'Result'}")
    print("-" * 25)
    for p, q in product([True, False], repeat=2):
        result = conjunction(disjunction(p, negation(q)), negation(p))
        print(f"{str(p):<8}{str(q):<8}{result}")


# -----------------------------------------------------------------------------
#  Task 20: Check Logical Equivalence - p->q  vs  ~p\/q
# -----------------------------------------------------------------------------
def check_equivalence_task20():
    print("\nTask 20: Logical Equivalence Check")
    print(f"{'p':<8}{'q':<8}{'p->q':<12}{'~p\/q':<12}{'Equal?'}")
    print("-" * 45)
    equivalent = True
    for p, q in product([True, False], repeat=2):
        col1 = implication(p, q)
        col2 = disjunction(negation(p), q)
        equal = col1 == col2
        if not equal:
            equivalent = False
        print(f"{str(p):<8}{str(q):<8}{str(col1):<12}{str(col2):<12}{equal}")
    print("\nVerdict:", "Logically Equivalent" if equivalent else "Not Equivalent")


# -----------------------------------------------------------------------------
#  Task 21: Check Tautology - (p->q) \/ (q->p)
# -----------------------------------------------------------------------------
def check_tautology_task21():
    print("\nTask 21: Tautology Check for (p->q) \/ (q->p)")
    all_true = True
    for p, q in product([True, False], repeat=2):
        result = disjunction(implication(p, q), implication(q, p))
        if not result:
            all_true = False
    print("Result:", "Tautology [OK]" if all_true else "Not a Tautology")


# -----------------------------------------------------------------------------
#  Task 22: Check if (p /\ ~p) is Always False
# -----------------------------------------------------------------------------
def check_contradiction_task22():
    print("\nTask 22: Contradiction Check for (p /\ ~p)")
    all_false = True
    for p in [True, False]:
        result = conjunction(p, negation(p))
        if result:
            all_false = False
    print("Result:", "Contradiction (always False) [OK]" if all_false
          else "Not always False")


# -----------------------------------------------------------------------------
#  Task 23: Classify Expression - Tautology / Contradiction / Contingency
# -----------------------------------------------------------------------------
def classify_expression(expr_func, variables_count):
    """Classify a boolean function as Tautology, Contradiction, or Contingency."""
    results = [expr_func(*combo)
               for combo in product([True, False], repeat=variables_count)]
    if all(results):
        return "Tautology"
    if not any(results):
        return "Contradiction"
    return "Contingency"

def task23():
    print("\nTask 23: Classify (p \/ q) /\ (~p \/ ~q)")
    def expr(p, q):
        return conjunction(disjunction(p, q),
                           disjunction(negation(p), negation(q)))
    label = classify_expression(expr, 2)
    print("Classification:", label)


# -----------------------------------------------------------------------------
#  Task 24: Check if r is Logically Entailed by {p->q, q->r, p}
# -----------------------------------------------------------------------------
def check_entailment_task24():
    print("\nTask 24: Entailment Check  KB = {p->q, q->r, p}  |= r ?")
    entailed = True
    for p, q, r in product([True, False], repeat=3):
        kb = implication(p, q) and implication(q, r) and p
        if kb and not r:
            entailed = False
    print("Entailed:", entailed)


# -----------------------------------------------------------------------------
#  Task 25: Simple Model Checker
# -----------------------------------------------------------------------------
def model_checker(kb_formulas, query_formula, variables):
    """
    kb_formulas : list of callables (truth-value functions of *variables)
    query_formula: callable
    variables    : number of propositional variables
    Returns True if KB entails query.
    """
    for combo in product([True, False], repeat=variables):
        kb_true = all(f(*combo) for f in kb_formulas)
        if kb_true and not query_formula(*combo):
            return False
    return True

def task25():
    print("\nTask 25: Model Checker")
    # KB: p->q, q->r; Query: p->r
    kb = [lambda p, q, r: implication(p, q),
          lambda p, q, r: implication(q, r)]
    query = lambda p, q, r: implication(p, r)
    result = model_checker(kb, query, 3)
    print("KB entails query:", result)


# -----------------------------------------------------------------------------
#  Task 26: Auto Truth Table from Expression String
# -----------------------------------------------------------------------------
def auto_truth_table(expression_str, variable_names):
    """Evaluates and prints truth table for an expression given as a string."""
    import re
    print(f"\nTask 26: Auto Truth Table for: {expression_str}")
    header = "  ".join(f"{v:<6}" for v in variable_names) + f"  {'Result'}"
    print(header)
    print("-" * len(header))
    for combo in product([True, False], repeat=len(variable_names)):
        env = dict(zip(variable_names, combo))
        # Replace logical words for Python evaluation
        expr = expression_str
        expr = expr.replace('not ', 'not ').replace('and', 'and').replace('or','or')
        result = eval(expr, {"__builtins__": {}}, env)
        row = "  ".join(f"{str(v):<6}" for v in combo) + f"  {result}"
        print(row)


# -----------------------------------------------------------------------------
#  Task 27: Convert p -> (q /\ r) to CNF
# -----------------------------------------------------------------------------
def cnf_task27():
    """Convert p -> (q /\ r) to CNF programmatically."""
    print("\nTask 27: CNF Conversion for p -> (q /\ r)")
    print("Original : p -> (q /\ r)")
    print("Step 1   : Eliminate ->  ->  ~p \/ (q /\ r)")
    print("Step 2   : Distribute \/ over /\:")
    print("           (~p \/ q) /\ (~p \/ r)")
    print("CNF      : (~p \/ q) /\ (~p \/ r)")
    print()
    # Verify with truth table
    print(f"{'p':<8}{'q':<8}{'r':<8}{'Original':<12}{'CNF'}")
    print("-" * 45)
    for p, q, r in product([True, False], repeat=3):
        original = implication(p, conjunction(q, r))
        cnf = conjunction(disjunction(not p, q), disjunction(not p, r))
        print(f"{str(p):<8}{str(q):<8}{str(r):<8}{str(original):<12}{cnf}"
              + ("  [OK]" if original == cnf else "  [FAIL]"))


# -----------------------------------------------------------------------------
#  Task 28: Dynamic Truth Table - Auto-Detect Variables
# -----------------------------------------------------------------------------
def dynamic_truth_table(expression_str):
    """Detects variables in expression and generates truth table dynamically."""
    import re
    # Extract unique single-letter variable names (skip Python keywords)
    keywords = {'and', 'or', 'not', 'True', 'False', 'if', 'else'}
    tokens = re.findall(r'\b[a-z]\b', expression_str)
    variables = sorted(set(tokens) - keywords)
    n = len(variables)
    print(f"\nTask 28: Dynamic Truth Table")
    print(f"Expression : {expression_str}")
    print(f"Detected variables ({n}): {variables}")
    header = "  ".join(f"{v:<7}" for v in variables) + "  Result"
    print(header)
    print("-" * len(header))
    for combo in product([True, False], repeat=n):
        env = dict(zip(variables, combo))
        result = eval(expression_str, {"__builtins__": {}}, env)
        row = "  ".join(f"{str(v):<7}" for v in combo) + f"  {result}"
        print(row)


# =============================================================================
#  MAIN - Run All Tasks
# =============================================================================
if __name__ == "__main__":

    # ── Example graphs ──────────────────────────────────────────────────────
    graph_uw = {          # unweighted adjacency list
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [],
        'E': [],
        'F': [],
        'G': []
    }

    graph_w = {           # weighted adjacency list  {node: [(nb, cost)]}
        'A': [('B', 1), ('C', 4)],
        'B': [('D', 2), ('E', 5)],
        'C': [('F', 1), ('G', 3)],
        'D': [('G', 8)],
        'E': [('G', 2)],
        'F': [('G', 1)],
        'G': []
    }

    heuristic = {'A': 6, 'B': 4, 'C': 3, 'D': 5, 'E': 2, 'F': 1, 'G': 0}

    # ── Part A ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print(" PART A - UNINFORMED SEARCH")
    print("=" * 55)

    print("\n[BFS Traversal]")
    print(" ".join(bfs(graph_uw, 'A')))

    print("\n[DFS Recursive Traversal]")
    print(" ".join(dfs(graph_uw, 'A')))

    print("\n[DFS Iterative Traversal]")
    print(" ".join(dfs_iterative(graph_uw, 'A')))

    print("\n[Task 7 - BFS Shortest Path A -> G]")
    path = bfs_shortest_path(graph_uw, 'A', 'G')
    print(path)

    print("\n[Tasks 8 & 9 - BFS on 2D Grid]")
    grid = [
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 0]
    ]
    g_path, g_len = bfs_grid(grid, (0, 0), (3, 3))
    print(f"Path  : {g_path}")
    print(f"Length: {g_len} steps")

    print("\n[Task 10 - Cycle Detection]")
    cyclic_graph    = {'A': ['B'], 'B': ['C'], 'C': ['A']}
    acyclic_graph   = {'A': ['B'], 'B': ['C'], 'C': []}
    print("Cyclic graph has cycle:", has_cycle(cyclic_graph))
    print("Acyclic graph has cycle:", has_cycle(acyclic_graph))

    print("\n[Task 11 - Topological Sort]")
    dag = {'A': ['C', 'B'], 'B': ['D'], 'C': ['B'], 'D': []}
    print(topological_sort(dag))

    print("\n[UCS - A to G]")
    ucs_path, ucs_cost, ucs_exp = ucs(graph_w, 'A', 'G')
    print(f"Path: {ucs_path}  Cost: {ucs_cost}  Expanded: {ucs_exp}")

    print("\n[IDDFS - A to G]")
    print(iddfs(graph_uw, 'A', 'G'))

    # ── Part B ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" PART B - INFORMED SEARCH")
    print("=" * 55)

    print("\n[Greedy Best-First - A to G]")
    gbfs_path, gbfs_exp = greedy_best_first(graph_uw, 'A', 'G', heuristic)
    print(f"Path: {gbfs_path}  Expanded: {gbfs_exp}")

    print("\n[A* Search - A to G]")
    as_path, as_cost, as_exp = a_star(graph_w, 'A', 'G', heuristic)
    print(f"Path: {as_path}  Cost: {as_cost}  Expanded: {as_exp}")

    print("\n[Task 3 - A* for 8-Puzzle]")
    start_puzzle = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    goal_puzzle  = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    puz_path, puz_moves, puz_exp = a_star_puzzle(start_puzzle, goal_puzzle)
    print(f"Moves to solve: {puz_moves}  Nodes expanded: {puz_exp}")

    print("\n[Task 12 - Next States of 8-Puzzle]")
    init = ((1,2,3),(4,0,6),(7,5,8))
    next_s = generate_next_states(init, visited_states={init})
    print(f"{len(next_s)} next states generated (excluding current):")
    for s in next_s:
        print(list(list(r) for r in s))

    print("\n[Task 13 - Compare All Search Algorithms]")
    compare_search_algorithms(graph_uw, graph_w, heuristic, 'A', 'G')

    print("\n[Task 14 - Game Tree: All Root-to-Leaf Paths]")
    game_tree = {
        'root': ['A', 'B'],
        'A':    ['A1', 'A2'],
        'B':    ['B1'],
        'A1':   [],
        'A2':   [],
        'B1':   ['B1a', 'B1b'],
        'B1a':  [],
        'B1b':  []
    }
    print_game_tree_paths(game_tree, 'root')

    print("\n[Task 15 - A* on 2D Grid (Manhattan Distance)]")
    a_star_grid(grid, (0, 0), (3, 3))

    # ── Part C ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(" PART C - PROPOSITIONAL LOGIC")
    print("=" * 55)

    truth_table_task04()
    truth_table_task16()
    truth_table_task17()
    truth_table_task18()
    truth_table_task19()
    check_equivalence_task20()
    check_tautology_task21()
    check_contradiction_task22()
    task23()
    check_entailment_task24()
    task25()
    auto_truth_table("p and not q", ['p', 'q'])
    cnf_task27()
    dynamic_truth_table("(p or q) and (not r or p)")

