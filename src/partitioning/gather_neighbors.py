from collections import deque
import networkx as nx

def nx_gather(G, sources, distance=None):
    if sources in G:
        sources = [sources]

    queue = deque(sources)
    depths = deque([0 for _ in queue])
    visited = set(sources)

    for source in queue:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    while queue:
        node = queue[0]
        depth = depths[0]

        if distance is not None and depth > distance: return

        yield queue[0]

        queue.popleft()
        depths.popleft()

        for child in G[node]:
            if child not in visited:
                visited.add(child)
                queue.append(child)
                depths.append(depth + 1)