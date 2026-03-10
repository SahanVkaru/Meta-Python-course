
import heapq

def uniform_cost_search(graph, start, goal):
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    while pq:
        curr_cost, node = heapq.heappop(pq)
        if node == goal:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            return path[::-1]
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node]:
                new_cost = curr_cost + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    parent[neighbor] = node
    return None

weighted_graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('A', 2), ('D', 3), ('E', 1)],
    'C': [('A', 5), ('F', 4)],
    'D': [('B', 3)],
    'E': [('B', 1), ('G', 6)],
    'F': [('C', 4)],
    'G': []
}

print("UCS Path:", uniform_cost_search(weighted_graph, 'A', 'G'))