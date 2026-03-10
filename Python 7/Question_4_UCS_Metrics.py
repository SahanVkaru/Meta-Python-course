import time
import heapq

def ucs_with_metrics(graph, start, goal):
    start_time = time.time()
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost = {start: 0}
    nodes_expanded = 0
    
    while pq:
        curr_cost, node = heapq.heappop(pq)
        nodes_expanded += 1
        
        if node == goal:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            end_time = time.time()
            return {
                "path": path[::-1],
                "total_cost": curr_cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": end_time - start_time
            }
        
        if node not in visited:
            visited.add(node)
            for neighbor, weight in graph[node]:
                new_cost = curr_cost + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    parent[neighbor] = node
    
    end_time = time.time()
    return None

# Weighted graph
weighted_graph = {
    'A': [('B',1), ('C',3)],
    'B': [('D',1), ('E',5)],
    'C': [('F',2)],
    'D': [],
    'E': [('G',1)],
    'F': [],
    'G': []
}

result = ucs_with_metrics(weighted_graph, 'A', 'G')

print("UCS Results:")
print(f"Path: {result['path']}")
print(f"Total cost: {result['total_cost']}")
print(f"Nodes expanded: {result['nodes_expanded']}")
print(f"Execution time: {result['execution_time']:.6f} seconds")