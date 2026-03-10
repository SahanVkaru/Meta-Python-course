def dfs_grid(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    stack = [(start, [start])]
    visited = set([start])
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    
    while stack:
        (x, y), path = stack.pop()
        if (x, y) == goal:
            return path
        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    stack.append(((nx, ny), path + [(nx, ny)]))
    return None

grid = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
]

print("DFS Grid Path:", dfs_grid(grid, (0,0), (3,3)))