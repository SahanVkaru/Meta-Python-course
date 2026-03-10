def dfs_path(graph, start, goal, path=None, visited=None):
 if visited is None:
  visited = set()
 if path is None:
  path = []
 visited.add(start)
 path = path + [start]
 if start == goal:
  return path
 for neighbor in graph[start]:
  if neighbor not in visited:
   result = dfs_path(graph, neighbor, goal, path, visited)
   if result:
    return result
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
print("DFS Path:", dfs_path(graph, 'A', 'G'))
