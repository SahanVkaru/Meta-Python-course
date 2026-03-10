import time
import heapq
from collections import deque

def bfs(graph, start, goal):
    start_time = time.time()
    visited = set()
    queue = deque([(start, [start])])
    nodes_expanded = 0
    
    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1
        
        if node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": "N/A",
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, {}):
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs(graph, start, goal):
    start_time = time.time()
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0
    
    while stack:
        node, path = stack.pop()
        nodes_expanded += 1
        
        if node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": "N/A",
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(list(graph.get(node, {}).keys())):
                stack.append((neighbor, path + [neighbor]))
    return None

def greedy(graph, start, goal, heuristic):
    start_time = time.time()
    visited = set()
    pq = [(heuristic[start], start, [start], 0)]
    nodes_expanded = 0
    
    while pq:
        _, node, path, cost = heapq.heappop(pq)
        nodes_expanded += 1
        
        if node in visited:
            continue
        visited.add(node)
        
        if node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor], cost + weight))
    return None

def a_star(graph, start, goal, heuristic):
    start_time = time.time()
    open_list = []
    heapq.heappush(open_list, (0, start, [start], 0))
    visited = set()
    nodes_expanded = 0
    
    while open_list:
        f_cost, current, path, g_cost = heapq.heappop(open_list)
        nodes_expanded += 1
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": g_cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        for neighbor, cost in graph.get(current, {}).items():
            if neighbor not in visited:
                new_g_cost = g_cost + cost
                new_f_cost = new_g_cost + heuristic[neighbor]
                heapq.heappush(open_list, (new_f_cost, neighbor, path + [neighbor], new_g_cost))
    return None

def ucs(graph, start, goal):
    start_time = time.time()
    pq = [(0, start, [start])]
    visited = set()
    nodes_expanded = 0
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
    return None

# Graph and heuristic definition
graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'D': 1, 'E': 5},
    'C': {'F': 2},
    'D': {},
    'E': {'G': 1},
    'F': {},
    'G': {}
}

heuristic = {
    'A': 7, 'B': 6, 'C': 2, 'D': 6, 'E': 1, 'F': 3, 'G': 0
}

start_node = 'A'
goal_node = 'G'

# Run all algorithms
algorithms = {
    "BFS": bfs(graph, start_node, goal_node),
    "DFS": dfs(graph, start_node, goal_node),
    "Greedy": greedy(graph, start_node, goal_node, heuristic),
    "A*": a_star(graph, start_node, goal_node, heuristic),
    "UCS": ucs(graph, start_node, goal_node)
}

# Display results in formatted table
print(f"{'Algorithm':<10} | {'Path':<15} | {'Length':<6} | {'Cost':<6} | {'Expanded':<8} | {'Time (s)':<10}")
print("-" * 75)

for name, result in algorithms.items():
    if result:
        path_str = "->".join(result['path'])
        cost = result['total_cost'] if result['total_cost'] != "N/A" else "N/A"
        print(f"{name:<10} | {path_str:<15} | {result['path_length']:<6} | {cost:<6} | {result['nodes_expanded']:<8} | {result['execution_time']:.6f}")
    else:
        print(f"{name:<10} | {'No path found':<15} | {'N/A':<6} | {'N/A':<6} | {'N/A':<8} | {'N/A'}")