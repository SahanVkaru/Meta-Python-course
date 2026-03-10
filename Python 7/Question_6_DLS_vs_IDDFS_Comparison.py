def depth_limited_dfs(graph, start, goal, limit, path=None):
    if path is None:
        path = [start]
    
    nodes_expanded = 1
    
    if start == goal:
        return path, nodes_expanded, limit
    
    if limit <= 0:
        return None, nodes_expanded, limit
    
    for neighbor in graph[start]:
        if neighbor not in path:
            result, expanded, _ = depth_limited_dfs(
                graph,
                neighbor,
                goal,
                limit-1,
                path + [neighbor]
            )
            nodes_expanded += expanded
            if result:
                return result, nodes_expanded, limit
    
    return None, nodes_expanded, limit

def iterative_deepening_dfs(graph, start, goal):
    total_nodes_expanded = 0
    max_depth = len(graph)
    
    for depth in range(max_depth + 1):
        result, expanded, _ = depth_limited_dfs(graph, start, goal, depth)
        total_nodes_expanded += expanded
        
        print(f"Depth {depth}: Nodes expanded = {expanded}")
        
        if result:
            return {
                "path": result,
                "solution_depth": depth,
                "total_nodes_expanded": total_nodes_expanded
            }
    
    return {
        "path": None,
        "solution_depth": -1,
        "total_nodes_expanded": total_nodes_expanded
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

# Test Depth-Limited DFS with different limits
print("Depth-Limited DFS Results:")
for limit in range(1, 5):
    result, expanded, _ = depth_limited_dfs(graph, 'A', 'G', limit)
    print(f"  Limit {limit}: Path = {result}, Nodes expanded = {expanded}")

print("\nIterative Deepening DFS Results:")
iddfs_result = iterative_deepening_dfs(graph, 'A', 'G')
print(f"Solution found at depth: {iddfs_result['solution_depth']}")
print(f"Total nodes expanded: {iddfs_result['total_nodes_expanded']}")
print(f"Path: {iddfs_result['path']}")