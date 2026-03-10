import time
from collections import deque

def bfs_grid(grid, start, goal):
    start_time = time.time()
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        if (r, c) == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_length": len(path) - 1,
                "visited_cells": len(visited),
                "execution_time": execution_time
            }
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                queue.append(((nr, nc), new_path))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "path_length": -1,
        "visited_cells": len(visited),
        "execution_time": execution_time
    }

def dfs_grid(grid, start, goal):
    start_time = time.time()
    rows, cols = len(grid), len(grid[0])
    stack = [(start, [start])]
    visited = {start}
    
    while stack:
        (r, c), path = stack.pop()
        
        if (r, c) == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return {
                "path": path,
                "path_length": len(path) - 1,
                "visited_cells": len(visited),
                "execution_time": execution_time
            }
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                stack.append(((nr, nc), new_path))
                
    end_time = time.time()
    execution_time = end_time - start_time
    return {
        "path": None,
        "path_length": -1,
        "visited_cells": len(visited),
        "execution_time": execution_time
    }

# 2D grid with obstacles (0 = free, 1 = obstacle)
grid = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
]

start_pos = (0, 0)
goal_pos = (3, 3)

bfs_results = bfs_grid(grid, start_pos, goal_pos)
dfs_results = dfs_grid(grid, start_pos, goal_pos)

print("BFS Grid Results:")
if bfs_results["path"]:
    print(f"  Path: {bfs_results['path']}")
    print(f"  Path length: {bfs_results['path_length']}")
    print(f"  Visited cells: {bfs_results['visited_cells']}")
    print(f"  Execution time: {bfs_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nDFS Grid Results:")
if dfs_results["path"]:
    print(f"  Path: {dfs_results['path']}")
    print(f"  Path length: {dfs_results['path_length']}")
    print(f"  Visited cells: {dfs_results['visited_cells']}")
    print(f"  Execution time: {dfs_results['execution_time']:.6f} seconds")
else:
    print("  No path found.")

print("\nComparison:")
if bfs_results["path"] and dfs_results["path"]:
    if bfs_results["path_length"] < dfs_results["path_length"]:
        print("BFS found a shorter path.")
    elif dfs_results["path_length"] < bfs_results["path_length"]:
        print("DFS found a shorter path.")
    else:
        print("Both algorithms found paths of the same length.")

    if bfs_results["visited_cells"] < dfs_results["visited_cells"]:
        print("BFS visited fewer cells.")
    elif dfs_results["visited_cells"] < bfs_results["visited_cells"]:
        print("DFS visited fewer cells.")
    else:
        print("Both algorithms visited the same number of cells.")
else:
    print("Could not compare paths as one or both algorithms did not find a path.")
