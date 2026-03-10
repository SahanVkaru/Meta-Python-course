import heapq

def a_star(graph, start, goal, heuristic):
    open_list = []
    heapq.heappush(open_list, (0, start))
    g_cost = {start: 0}
    parent = {start: None}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]
        
        for neighbor, cost in graph[current]:
            new_cost = g_cost[current] + cost
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))
                parent[neighbor] = current
    
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

heuristic = {
    'A': 7, 'B': 6, 'C': 2, 'D': 6, 'E': 1, 'F': 3, 'G': 0
}

print("A* Path:", a_star(weighted_graph, 'A', 'G', heuristic))