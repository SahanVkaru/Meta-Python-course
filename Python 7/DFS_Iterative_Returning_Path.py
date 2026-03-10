def dfs_iterative_path(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph[node]):
                stack.append((neighbor, path + [neighbor]))
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

print("DFS Iterative Path:", dfs_iterative_path(graph, 'A', 'G'))