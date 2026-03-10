import time
from collections import deque

def bfs_shortest_path_metrics(graph, start, goal):
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
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append((neighbor, path + [neighbor]))
    
    end_time = time.time()
    return {
        "path": None,
        "path_length": 0,
        "nodes_expanded": nodes_expanded,
        "execution_time": end_time - start_time
    }

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': [],
    'G': []
}

result = bfs_shortest_path_metrics(graph, 'A', 'G')

print("Path:", result["path"])
print("Path Length:", result["path_length"])
print("Number of nodes expanded:", result["nodes_expanded"])
print(f"Execution time: {result['execution_time']:.6f} seconds")