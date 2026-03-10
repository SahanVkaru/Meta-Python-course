import time
import heapq

def ucs(graph, start, goal):
    start_time = time.time()
    nodes_expanded = 0
    
    # Priority queue stores tuples of (cost, node, path)
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
            execution_time = end_time - start_time
            return {
                "path": path,
                "total_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }
            
        for neighbor, weight in graph.get(current_node, []):
            if neighbor not in visited:
                new_cost = cost + weight
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (new_cost, neighbor, new_path))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "total_cost": -1,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }

# Weighted graph
weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

start_node = 'A'
goal_node = 'D'

results = ucs(weighted_graph, start_node, goal_node)

if results["path"]:
    print(f"Path: {results['path']}")
    print(f"Total cost: {results['total_cost']}")
    print(f"Nodes expanded: {results['nodes_expanded']}")
    print(f"Execution time: {results['execution_time']:.6f} seconds")
else:
    print("No path found.")
