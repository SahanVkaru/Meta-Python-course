def has_cycle(graph):
    visited = set()
    recursion_stack = set()
    
    def check_cycle(node):
        visited.add(node)
        recursion_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if check_cycle(neighbor):
                    return True
            elif neighbor in recursion_stack:
                return True
                
        recursion_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if check_cycle(node):
                return True
    return False

def topological_sort(graph):
    if has_cycle(graph):
        print("Topological sorting not possible due to a cycle.")
        return None
        
    visited = set()
    stack = []
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)
        
    for node in graph:
        if node not in visited:
            dfs(node)
            
    return stack[::-1]

# Graph with a cycle
graph_with_cycle = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']
}

# Graph without a cycle (DAG)
dag = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}

print("Testing with a cyclic graph:")
result_cycle = topological_sort(graph_with_cycle)
if result_cycle:
    print(f"  Topological Sort: {result_cycle}")

print("\nTesting with a DAG:")
result_dag = topological_sort(dag)
if result_dag:
    print(f"  Topological Sort: {result_dag}")
