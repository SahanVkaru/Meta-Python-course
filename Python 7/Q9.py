import time
import heapq
from collections import deque

# --- Algorithm Implementations ---

def bfs(graph, start, goal):
    start_time = time.time()
    nodes_expanded = 0
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current_node, path = queue.popleft()
        nodes_expanded += 1
        
        if current_node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": "N/A",
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
            
        for neighbor in graph.get(current_node, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    return None

def dfs(graph, start, goal):
    start_time = time.time()
    nodes_expanded = 0
    stack = [(start, [start])]
    visited = {start}
    
    while stack:
        current_node, path = stack.pop()
        nodes_expanded += 1
        
        if current_node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": "N/A",
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
            
        for neighbor in reversed(list(graph.get(current_node, {}).keys())):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
                
    return None

def ucs(graph, start, goal):
    start_time = time.time()
    nodes_expanded = 0
    pq = [(0, start, [start])]
    visited = set()
    
    while pq:
        cost, current_node, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
            
        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))
                
    return None

def greedy_search(graph, start, goal, heuristic):
    start_time = time.time()
    nodes_expanded = 0
    pq = [(heuristic[start], start, [start], 0)]
    visited = set()
    
    while pq:
        _, current_node, path, cost = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
            
        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor, path + [neighbor], cost + weight))
                
    return None

def a_star(graph, start, goal, heuristic):
    start_time = time.time()
    nodes_expanded = 0
    pq = [(heuristic[start], 0, start, [start])]
    visited = set()
    
    while pq:
        _, cost, current_node, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        if current_node == goal:
            end_time = time.time()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
            
        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor not in visited:
                new_cost = cost + weight
                heapq.heappush(pq, (new_cost + heuristic[neighbor], new_cost, neighbor, path + [neighbor]))
                
    return None

# --- Main Execution ---

# Sample graph, start, goal, and heuristic
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'D': 5, 'E': 2},
    'C': {'F': 3},
    'D': {},
    'E': {'F': 1},
    'F': {}
}
start_node = 'A'
goal_node = 'F'
heuristic = {'A': 5, 'B': 3, 'C': 2, 'D': 4, 'E': 1, 'F': 0}

# Run all algorithms
results = {
    "BFS": bfs(graph, start_node, goal_node),
    "DFS": dfs(graph, start_node, goal_node),
    "UCS": ucs(graph, start_node, goal_node),
    "Greedy": greedy_search(graph, start_node, goal_node, heuristic),
    "A*": a_star(graph, start_node, goal_node, heuristic)
}

# Display results in a formatted table
print(f"{'Algorithm':<10} | {'Path':<20} | {'Length':<7} | {'Cost':<7} | {'Expanded':<9} | {'Time (s)':<10}")
print("-" * 80)

for name, res in results.items():
    if res:
        path_str = '->'.join(res['path'])
        print(f"{name:<10} | {path_str:<20} | {res['path_length']:<7} | {res['total_cost']:<7} | {res['nodes_expanded']:<9} | {res['execution_time']:.6f}")
    else:
        print(f"{name:<10} | {'No path found':<20} | {'N/A':<7} | {'N/A':<7} | {'N/A':<9} | {'N/A'}")
