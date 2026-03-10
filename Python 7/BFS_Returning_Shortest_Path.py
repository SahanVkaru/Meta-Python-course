from collections import deque

def bfs_shortest_path(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append((neighbor, path + [neighbor]))
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

print("Shortest Path:", bfs_shortest_path(graph, 'A', 'G'))