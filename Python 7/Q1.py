import time
from collections import deque

def bfs_shortest_path(graph, start, goal):
    start_time = time.time()
    
    nodes_expanded = 0
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current_node, path = queue.popleft()
        nodes_expanded += 1
        
        if current_node == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_length": len(path) - 1,
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }
            
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append((neighbor, new_path))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "path_length": -1,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }

# Sample graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E', 'G'],
    'G': ['F']
}

start_node = 'A'
goal_node = 'G'

results = bfs_shortest_path(graph, start_node, goal_node)

if results["path"]:
    print(f"Path: {results['path']}")
    print(f"Path length: {results['path_length']}")
    print(f"Number of nodes expanded: {results['nodes_expanded']}")
    print(f"Execution time: {results['execution_time']:.6f} seconds")
else:
    print("No path found.")
    print(f"Number of nodes expanded: {results['nodes_expanded']}")
    print(f"Execution time: {results['execution_time']:.6f} seconds")
