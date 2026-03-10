import heapq

def greedy_best_first(graph, start, goal, heuristic):
    visited = set()
    pq = [(heuristic[start], start)]
    parent = {start: None}
    
    while pq:
        _, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        
        if node == goal:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            return path[::-1]
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor))
                if neighbor not in parent:
                    parent[neighbor] = node
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

heuristic = {
    'A': 7, 'B': 6, 'C': 2, 'D': 6, 'E': 1, 'F': 3, 'G': 0
}

print("Greedy Best-First Path:", greedy_best_first(graph, 'A', 'G', heuristic))