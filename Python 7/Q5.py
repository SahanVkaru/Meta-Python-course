import time
import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    start_time = time.time()
    nodes_expanded = 0
    
    pq = [(heuristic[start], start, [start], 0)]  # (h_cost, node, path, g_cost)
    visited = set()
    
    while pq:
        _, current_node, path, cost = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }
            
        for neighbor, weight in graph.get(current_node, []):
            if neighbor not in visited:
                new_cost = cost + weight
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (heuristic[neighbor], neighbor, new_path, new_cost))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "path_cost": -1,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }

def a_star_search(graph, start, goal, heuristic):
    start_time = time.time()
    nodes_expanded = 0
    
    pq = [(heuristic[start], 0, start, [start])]  # (f_cost, g_cost, node, path)
    visited = set()
    
    while pq:
        _, cost, current_node, path = heapq.heappop(pq)
        nodes_expanded += 1
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_cost": cost,
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }
            
        for neighbor, weight in graph.get(current_node, []):
            if neighbor not in visited:
                new_cost = cost + weight
                f_cost = new_cost + heuristic[neighbor]
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (f_cost, new_cost, neighbor, new_path))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "path_cost": -1,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }

# Sample graph and heuristic
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}
heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 0}

start_node = 'A'
goal_node = 'D'

gbfs_results = greedy_best_first_search(graph, start_node, goal_node, heuristic)
astar_results = a_star_search(graph, start_node, goal_node, heuristic)

print("Greedy Best-First Search Results:")
if gbfs_results["path"]:
    print(f"  Path: {gbfs_results['path']}")
    print(f"  Path cost: {gbfs_results['path_cost']}")
    print(f"  Nodes expanded: {gbfs_results['nodes_expanded']}")
    print(f"  Execution time: {gbfs_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nA* Search Results:")
if astar_results["path"]:
    print(f"  Path: {astar_results['path']}")
    print(f"  Path cost: {astar_results['path_cost']}")
    print(f"  Nodes expanded: {astar_results['nodes_expanded']}")
    print(f"  Execution time: {astar_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nComparison:")
if gbfs_results["path"] and astar_results["path"]:
    if astar_results["path_cost"] < gbfs_results["path_cost"]:
        print("A* gives the optimal solution.")
    else:
        print("Greedy Best-First Search may not give the optimal solution.")

    if astar_results["execution_time"] < gbfs_results["execution_time"]:
        print("A* is faster.")
    else:
        print("Greedy Best-First Search is faster.")
else:
    print("Could not compare paths as one or both algorithms did not find a path.")
