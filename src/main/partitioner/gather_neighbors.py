from collections import deque

def descendants_at_distance_multisource(adj_list, sources, distance=None):
    queue = deque(sources)
    depths = deque([0 for _ in queue])
    visited = set(sources)

    while queue:
        node = queue[0]
        depth = depths[0]

        if distance is not None and depth > distance: return

        yield queue[0]

        queue.popleft()
        depths.popleft()

        for adj in adj_list[node]:
            if adj not in visited:
                visited.add(adj)
                queue.append(adj)
                depths.append(depth + 1)
                
def edge_boundary(adj_list, nbunch1):
    boundary = set()
    
    for u, adj in enumerate(adj_list):
        for v in adj:
            if u in nbunch1 and v not in nbunch1:
                boundary.add(u)
                
    return boundary