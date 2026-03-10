def dfs_all_paths(graph, start, path=None):
    """Print all root-to-leaf paths in a game tree"""
    if path is None:
        path = [start]
    
    if not graph[start]:  # Leaf node
        print("Path:", " -> ".join(path))
        return [path]
    
    all_paths = []
    for neighbor in graph[start]:
        paths = dfs_all_paths(graph, neighbor, path + [neighbor])
        all_paths.extend(paths)
    
    return all_paths

def count_leaf_nodes(graph):
    """Count total number of leaf nodes"""
    count = 0
    for node in graph:
        if not graph[node]:  # No children = leaf node
            count += 1
    return count

def max_depth(graph, start, current_depth=1):
    """Count maximum depth of the tree"""
    if not graph[start]:  # Leaf node
        return current_depth
    
    max_child_depth = 0
    for neighbor in graph[start]:
        child_depth = max_depth(graph, neighbor, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth

def count_connected_components(graph):
    """Return number of connected components in a disconnected graph"""
    visited = set()
    components = 0
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    for node in graph:
        if node not in visited:
            dfs(node)
            components += 1
    
    return components

# Game Tree
game_tree = {
    'S': ['A', 'B'],
    'A': ['C', 'D'],
    'B': ['E'],
    'C': [],
    'D': [],
    'E': []
}

print("1. All root-to-leaf paths in the game tree:")
all_paths = dfs_all_paths(game_tree, 'S')

print(f"\n2. Total number of leaf nodes: {count_leaf_nodes(game_tree)}")

print(f"\n3. Maximum depth of the tree: {max_depth(game_tree, 'S')}")

# Disconnected graph for testing connected components
disconnected_graph = {
    'A': ['B'],
    'B': ['A'],
    'C': ['D'],
    'D': ['C'],
    'E': [],
    'F': ['G'],
    'G': ['F']
}

print(f"\n4. Number of connected components in disconnected graph: {count_connected_components(disconnected_graph)}")

# Show the components
print("\nComponents visualization:")
visited = set()
component_num = 1

def show_component(graph, start, component_nodes):
    visited.add(start)
    component_nodes.append(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            show_component(graph, neighbor, component_nodes)

for node in disconnected_graph:
    if node not in visited:
        component_nodes = []
        show_component(disconnected_graph, node, component_nodes)
        print(f"  Component {component_num}: {component_nodes}")
        component_num += 1