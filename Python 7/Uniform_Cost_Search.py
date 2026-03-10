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
    'A': [('B',1), ('C',3)],
    'B': [('D',1), ('E',5)],
    'C': [('F',2)],
    'D': [],
    'E': [('G',1)],
    'F': [],
    'G': []
}

print("UCS Path:", uniform_cost_search(weighted_graph, 'A', 'G'))