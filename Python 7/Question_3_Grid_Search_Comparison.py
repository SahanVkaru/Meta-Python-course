from collections import deque

def bfs_grid(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = set([start])
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return {
                "path": path,
                "path_length": len(path) - 1,
                "visited_cells": len(visited)
            }
        
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    
    return None

def dfs_grid(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    stack = [(start, [start])]
    visited = set([start])
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    
    while stack:
        (x, y), path = stack.pop()
        if (x, y) == goal:
            return {
                "path": path,
                "path_length": len(path) - 1,
                "visited_cells": len(visited)
            }
        
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append(((nx, ny), path + [(nx, ny)]))
    
    return None

# 2D grid with obstacles (0 = free, 1 = obstacle)
grid = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
]

# Run BFS
bfs_result = bfs_grid(grid, (0,0), (3,3))
print("BFS Results:")
print(f"  Path: {bfs_result['path']}")
print(f"  Path length: {bfs_result['path_length']}")
print(f"  Visited cells: {bfs_result['visited_cells']}")

# Run DFS
dfs_result = dfs_grid(grid, (0,0), (3,3))
print("\nDFS Results:")
print(f"  Path: {dfs_result['path']}")
print(f"  Path length: {dfs_result['path_length']}")
print(f"  Visited cells: {dfs_result['visited_cells']}")

# Comparison
print("\nComparison:")
if bfs_result["path_length"] < dfs_result["path_length"]:
    print("BFS finds shorter path")
elif dfs_result["path_length"] < bfs_result["path_length"]:
    print("DFS finds shorter path")
else:
    print("Both algorithms find paths of equal length")

if bfs_result["visited_cells"] < dfs_result["visited_cells"]:
    print("BFS visits fewer cells")
elif dfs_result["visited_cells"] < bfs_result["visited_cells"]:
    print("DFS visits fewer cells")
else:
    print("Both algorithms visit equal number of cells")