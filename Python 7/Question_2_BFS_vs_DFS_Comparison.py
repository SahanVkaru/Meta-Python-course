import time
from collections import deque

def bfs_with_metrics(graph, start, goal):
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
    return None

def dfs_with_metrics(graph, start, goal):
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
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph[node]):
                stack.append((neighbor, path + [neighbor]))
    
    end_time = time.time()
    return None

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': [],
    'G': []
}

# Run BFS
bfs_result = bfs_with_metrics(graph, 'A', 'G')
print("BFS Results:")
print(f"  Path: {bfs_result['path']}")
print(f"  Path length: {bfs_result['path_length']}")
print(f"  Nodes expanded: {bfs_result['nodes_expanded']}")
print(f"  Execution time: {bfs_result['execution_time']:.6f} seconds")

# Run DFS
dfs_result = dfs_with_metrics(graph, 'A', 'G')
print("\nDFS Results:")
print(f"  Path: {dfs_result['path']}")
print(f"  Path length: {dfs_result['path_length']}")
print(f"  Nodes expanded: {dfs_result['nodes_expanded']}")
print(f"  Execution time: {dfs_result['execution_time']:.6f} seconds")

# Comparison
print("\nComparison:")
if bfs_result["path_length"] < dfs_result["path_length"]:
    print("BFS finds shorter path")
elif dfs_result["path_length"] < bfs_result["path_length"]:
    print("DFS finds shorter path")
else:
    print("Both algorithms find paths of equal length")

if bfs_result["nodes_expanded"] < dfs_result["nodes_expanded"]:
    print("BFS expands fewer nodes")
elif dfs_result["nodes_expanded"] < bfs_result["nodes_expanded"]:
    print("DFS expands fewer nodes")
else:
    print("Both algorithms expand equal number of nodes")