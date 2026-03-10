def has_cycle(graph):
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

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

# Test with a graph that has no cycle (DAG)
dag = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

print("Testing with DAG:")
if has_cycle(dag):
    print("Topological sorting not possible")
else:
    result = topological_sort(dag)
    print(f"Topological Sort: {result}")

# Test with a graph that has a cycle
cyclic_graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']
}

print("\nTesting with cyclic graph:")
if has_cycle(cyclic_graph):
    print("Topological sorting not possible")
else:
    result = topological_sort(cyclic_graph)
    print(f"Topological Sort: {result}")