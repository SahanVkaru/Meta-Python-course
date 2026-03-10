def depth_limited_dfs(graph, start, goal, limit, path=None):
    if path is None:
        path = [start]
    
    if start == goal:
        return path
    
    if limit <= 0:
        return None
    
    for neighbor in graph[start]:
        if neighbor not in path:
            result = depth_limited_dfs(
                graph,
                neighbor,
                goal,
                limit-1,
                path + [neighbor]
            )
            if result:
                return result
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

print("Depth Limited DFS:", depth_limited_dfs(graph, 'A', 'G', 3))