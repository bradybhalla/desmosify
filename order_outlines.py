# turn the points of an outline into ordered paths

from collections import deque
from random import randint

# minimum nodes to keep a subgraph
#   30 is a good number because a path will need to have length 30 to
#   be turned into a Fourier Series
#   ^^^ this is outdated, now it just makes more points if there aren't enough
MIN_NODES = 10


# check if two points in an outline are adjacent in the graph representation
# determined by distance between points
def is_adj(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 < 3

# order the points of an outline into a list of cycles


def order_outlines(outline_set):
    # turn into a graph
    points = list(outline_set)

    n = len(points)

    edges = [
        [j for j in range(n) if j != i and is_adj(points[i], points[j])]
        for i in range(n)
    ]

    # break into subgraphs
    disjoint_subgraphs = [
        i for i in get_disjoint_subgraphs(set(range(n)), edges)]

    # remove lines and articulation points
    clean_subgraphs = []
    for subgraph in disjoint_subgraphs:
        clean_subgraphs.extend(clean_up_subgraph(subgraph, edges))

    # find a long cycle in each subgraph
    orderings = []
    for ind, cs in enumerate(clean_subgraphs):
        path = get_ordering(cs, edges)
        if path is not None:
            orderings.append(path)

    return [[points[i] for i in p] for p in orderings]

# find disjoint subgraphs in nodes
# returns subgraphs found (a partition of nodes)


def get_disjoint_subgraphs(nodes, edges):
    n = len(edges)

    subsets = [0 for _ in range(n)]
    counter = 1

    while True:
        start = 0
        while start < n:
            if subsets[start] == 0:
                if start in nodes:
                    break
            start += 1
        if start == n:
            break

        queue = deque()
        queue.appendleft(start)

        subsets[start] = counter

        while len(queue) > 0:
            curr = queue.pop()

            for i in edges[curr]:
                if subsets[i] == 0:
                    if i not in nodes:
                        continue
                    queue.appendleft(i)
                    subsets[i] = counter

        counter += 1

    return [[j for j in range(n) if subsets[j] == i+1] for i in range(max(subsets))]

# clean up subgraph
# remove nodes of degree 1
# resolve articulation points
# returns nodes of clean subgraphs
# assume input nodes are all connected


def clean_up_subgraph(nodes, edges):
    while True:
        singles = False

        edge_counts = {i: sum([1 for j in edges[i] if j in nodes])
                       for i in nodes}
        for i in edge_counts:
            if edge_counts[i] < 2:
                nodes.remove(i)
                singles = True

        if not singles:
            break

    if len(nodes) < MIN_NODES:
        return []

    articulation, split_subgraphs = find_split_articulation(nodes, edges)
    if articulation is None:
        return split_subgraphs

    else:
        new_subgraphs = []
        for i in split_subgraphs:
            new_subgraphs.extend(clean_up_subgraph(i, edges))
        return new_subgraphs

# find articulation points in nodes and return subgraphs formed by
# splitting the articulation point


def find_split_articulation(nodes, edges):
    orig = get_disjoint_subgraphs(nodes, edges)
    if len(orig) > 1:
        raise ValueError("Nodes are not connected to begin with")

    nodes_set = set(nodes)
    articulation = None
    for i in nodes:
        nodes_set.remove(i)

        subgraphs = get_disjoint_subgraphs(nodes_set, edges)
        if len(subgraphs) > 1:
            articulation = i

        nodes_set.add(i)

        if articulation is not None:
            break

    if articulation is None:
        return (None, [list(nodes_set)])

    for i in subgraphs:
        i.append(articulation)
    return (articulation, subgraphs)

# depth first search with picking random edges


def random_dfs(start, target, nodes, edges, edges_deleted):
    visited = set()
    visited.add(start)

    stack = [start]

    while len(stack) > 0:
        curr = stack[-1]
        if curr == target:
            return stack

        unvisited_adj = []
        for adj in edges[curr]:
            if adj not in nodes:
                continue
            if adj in visited:
                continue
            if (curr, adj) in edges_deleted or (adj, curr) in edges_deleted:
                continue

            unvisited_adj.append(adj)

        if len(unvisited_adj) == 0:
            stack.pop()

        else:
            next_node = unvisited_adj[randint(0, len(unvisited_adj) - 1)]
            visited.add(next_node)
            stack.append(next_node)

    raise ValueError("target not found")

# order nodes into a long cycle
# runs a randomized DFS lots of times and uses the longest cycle found
# run until <percent_in_cycle> of nodes are in cycle or <max_trials> times


def get_ordering(nodes, edges, max_trials=2000, percent_in_cycle=0.9):
    possible_starts = []
    for i in nodes:
        adj = [j for j in edges[i] if j in nodes]
        if len(adj) == 2:
            possible_starts.append((i, adj[randint(0, 1)]))

    if len(possible_starts) == 0:
        print("No starting point for cycle found")
        return

    paths = []
    for i in range(max_trials):
        start, target = possible_starts[randint(0, len(possible_starts) - 1)]
        p = random_dfs(start, target, nodes, edges, [(start, target)])
        if len(p) > percent_in_cycle*len(nodes):
            return p
        paths.append(p)
    return max(paths, key=lambda x: len(x))
