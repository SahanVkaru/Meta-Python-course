def dfs_with_metrics(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0
    
    while stack:
        node, path = stack.pop()
        nodes_expanded += 1
        
        if node == goal:
            return path, nodes_expanded
        
        if node not in visited:
            visited.add(node)
            for neighbor in reversed(graph[node]):
                stack.append((neighbor, path + [neighbor]))
    
    return None, nodes_expanded

graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['G'],
    'F': [],
    'G': []
}

path, count = dfs_with_metrics(graph, 'A', 'G')
print("Path:", path)
print("Nodes Expanded:", count)