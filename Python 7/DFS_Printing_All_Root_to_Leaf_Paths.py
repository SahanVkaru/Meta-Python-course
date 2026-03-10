def dfs_all_paths(graph, start, path=None):
    if path is None:
        path = [start]
    
    if not graph[start]:
        print("Path:", path)
        return
    
    for neighbor in graph[start]:
        dfs_all_paths(graph, neighbor, path + [neighbor])

game_tree = {
    'S': ['A', 'B'],
    'A': ['C', 'D'],
    'B': ['E'],
    'C': [],
    'D': [],
    'E': []
}

dfs_all_paths(game_tree, 'S')