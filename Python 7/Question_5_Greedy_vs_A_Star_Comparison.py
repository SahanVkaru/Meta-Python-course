import time
import heapq

def greedy_search_with_metrics(graph, start, goal, heuristic):
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
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                new_cost = cost + weight
                new_path = path + [neighbor]
                heapq.heappush(pq, (heuristic[neighbor], neighbor, new_path, new_cost))
    
    return None

def a_star_with_metrics(graph, start, goal, heuristic):
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
                "path_cost": g_cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                new_g_cost = g_cost + cost
                new_f_cost = new_g_cost + heuristic[neighbor]
                new_path = path + [neighbor]
                heapq.heappush(open_list, (new_f_cost, neighbor, new_path, new_g_cost))
    
    return None

# Weighted graph and heuristic
graph = {
    'A': [('B',1), ('C',3)],
    'B': [('D',1), ('E',5)],
    'C': [('F',2)],
    'D': [],
    'E': [('G',1)],
    'F': [],
    'G': []
}

heuristic = {
    'A': 7, 'B': 6, 'C': 2, 'D': 6, 'E': 1, 'F': 3, 'G': 0
}

# Run Greedy Best-First Search
greedy_result = greedy_search_with_metrics(graph, 'A', 'G', heuristic)
print("Greedy Best-First Search Results:")
print(f"  Path: {greedy_result['path']}")
print(f"  Path cost: {greedy_result['path_cost']}")
print(f"  Nodes expanded: {greedy_result['nodes_expanded']}")
print(f"  Execution time: {greedy_result['execution_time']:.6f} seconds")

# Run A* Search
astar_result = a_star_with_metrics(graph, 'A', 'G', heuristic)
print("\nA* Search Results:")
print(f"  Path: {astar_result['path']}")
print(f"  Path cost: {astar_result['path_cost']}")
print(f"  Nodes expanded: {astar_result['nodes_expanded']}")
print(f"  Execution time: {astar_result['execution_time']:.6f} seconds")

# Comparison
print("\nComparison:")
if astar_result['path_cost'] <= greedy_result['path_cost']:
    print("A* gives optimal solution (guaranteed)")
else:
    print("Greedy found better solution (unexpected)")

if astar_result['execution_time'] < greedy_result['execution_time']:
    print("A* is faster")
else:
    print("Greedy is faster")