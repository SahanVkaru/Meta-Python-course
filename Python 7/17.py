def topological_sort(graph):
    visited = set()
    stack = []
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)
    for node in graph:
        if node not in visited:
            dfs(node)
    return stack[::-1]
dag = {
 'A': ['C'],
 'B': ['C', 'D'],
 'C': ['E'],
 'D': ['F'],
 'E': [],
 'F': []
}
print("Topological Order:", topological_sort(dag))