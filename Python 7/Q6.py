def depth_limited_dfs(graph, start, goal, limit):
    nodes_expanded = 0
    
    def dls_recursive(node, path, depth):
        nonlocal nodes_expanded
        nodes_expanded += 1
        
        if node == goal:
            return path
        
        if depth == 0:
            return None
            
        for neighbor in graph.get(node, []):
            if neighbor not in path:
                new_path = list(path)
                new_path.append(neighbor)
                result = dls_recursive(neighbor, new_path, depth - 1)
                if result:
                    return result
        return None

    result_path = dls_recursive(start, [start], limit)
    return result_path, nodes_expanded

def iterative_deepening_dfs(graph, start, goal):
    max_depth = len(graph)
    total_nodes_expanded = 0
    
    for depth in range(max_depth):
        path, nodes_expanded = depth_limited_dfs(graph, start, goal, depth)
        total_nodes_expanded += nodes_expanded
        if path:
            return {
                "solution_depth": depth,
                "nodes_expanded": total_nodes_expanded,
                "path": path
            }
            
    return {
        "solution_depth": -1,
        "nodes_expanded": total_nodes_expanded,
        "path": None
    }

# Sample graph
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

start_node = 'A'
goal_node = 'F'

# Test DLS with different depth limits
print("Depth-Limited DFS:")
for limit in range(1, 5):
    path, nodes_expanded = depth_limited_dfs(graph, start_node, goal_node, limit)
    print(f"  Limit {limit}: Path={path}, Nodes Expanded={nodes_expanded}")

# Test IDDFS
print("\nIterative Deepening DFS:")
iddfs_results = iterative_deepening_dfs(graph, start_node, goal_node)
if iddfs_results["path"]:
    print(f"  Solution found at depth: {iddfs_results['solution_depth']}")
    print(f"  Nodes expanded: {iddfs_results['nodes_expanded']}")
    print(f"  Path: {iddfs_results['path']}")
else:
    print("  No solution found.")
