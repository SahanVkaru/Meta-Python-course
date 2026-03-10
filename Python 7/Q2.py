import time
from collections import deque

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
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_length": len(path) - 1,
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }
            
        for neighbor in reversed(graph.get(current_node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                stack.append((neighbor, new_path))
                
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

bfs_results = bfs(graph, start_node, goal_node)
dfs_results = dfs(graph, start_node, goal_node)

print("BFS Results:")
if bfs_results["path"]:
    print(f"  Path: {bfs_results['path']}")
    print(f"  Path length: {bfs_results['path_length']}")
    print(f"  Nodes expanded: {bfs_results['nodes_expanded']}")
    print(f"  Execution time: {bfs_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nDFS Results:")
if dfs_results["path"]:
    print(f"  Path: {dfs_results['path']}")
    print(f"  Path length: {dfs_results['path_length']}")
    print(f"  Nodes expanded: {dfs_results['nodes_expanded']}")
    print(f"  Execution time: {dfs_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nComparison:")
if bfs_results["path"] and dfs_results["path"]:
    if bfs_results["path_length"] < dfs_results["path_length"]:
        print("BFS found a shorter path.")
    elif dfs_results["path_length"] < bfs_results["path_length"]:
        print("DFS found a shorter path.")
    else:
        print("Both algorithms found paths of the same length.")

    if bfs_results["nodes_expanded"] < dfs_results["nodes_expanded"]:
        print("BFS expanded fewer nodes.")
    elif dfs_results["nodes_expanded"] < bfs_results["nodes_expanded"]:
        print("DFS expanded fewer nodes.")
    else:
        print("Both algorithms expanded the same number of nodes.")
else:
    print("Could not compare paths as one or both algorithms did not find a path.")
