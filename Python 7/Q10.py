def find_all_paths(graph, start, path=[]):
    path = path + [start]
    if not graph.get(start):
        return [path]
    paths = []
    for node in graph[start]:
        new_paths = find_all_paths(graph, node, path)
        for new_path in new_paths:
            paths.append(new_path)
    return paths

def count_leaf_nodes(graph):
    count = 0
    for node in graph:
        if not graph.get(node):
            count += 1
    return count

def max_depth(graph, start):
    if not graph.get(start):
        return 1
    max_d = 0
    for node in graph[start]:
        d = max_depth(graph, node)
        if d > max_d:
            max_d = d
    return max_d + 1

def find_connected_components(graph):
    visited = set()
    components = []
    
    def dfs(node, current_component):
        visited.add(node)
        current_component.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, current_component)
                
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
            
    return components

# 1. Game Tree
game_tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

print("1. All root-to-leaf paths:")
all_paths = find_all_paths(game_tree, 'A')
for path in all_paths:
    print(f"  - {' -> '.join(path)}")

# 2. Count leaf nodes
leaf_count = count_leaf_nodes(game_tree)
print(f"\n2. Total number of leaf nodes: {leaf_count}")

# 3. Maximum depth
depth = max_depth(game_tree, 'A')
print(f"\n3. Maximum depth of the tree: {depth}")

# 4. Disconnected Graph
disconnected_graph = {
    'A': ['B'],
    'B': ['A'],
    'C': ['D'],
    'D': [],
    'E': ['F'],
    'F': []
}

components = find_connected_components(disconnected_graph)
print(f"\n4. Number of connected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  - Component {i+1}: {comp}")
