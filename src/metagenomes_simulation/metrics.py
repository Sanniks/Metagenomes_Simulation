import leidenalg

# --------------------------------------------------------------
# Network metrics (Using igraph networks)
# --------------------------------------------------------------


def number_of_nodes(graph):
    """Return the number of nodes.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        int: Number of nodes in the graph
    """
    return len(graph.vs)


def number_of_edges(graph):
    """Return the number of edges in the graph.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        int: Number of edges in the graph
    """
    return len(graph.es)


def average_degree(graph):
    """Return the average number of neighbours per node for undirected graph.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        float: Average degree of the graph
    """
    if graph.is_directed():
        raise ValueError("average_degree requires an undirected graph")

    n_nodes = number_of_nodes(graph)
    n_edges = number_of_edges(graph)

    if n_nodes == 0:
        return 0.0
    return 2 * n_edges / n_nodes


def network_density(graph):
    """Return the fraction of possible edges that are present.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        float: Network density
    """
    n_nodes = number_of_nodes(graph)
    n_edges = number_of_edges(graph)
    if n_nodes <= 1:
        return 0.0

    max_possible_edges = n_nodes * (n_nodes - 1) / 2
    return n_edges / max_possible_edges


def average_clustering(graph, weight_enabled=False):
    """Return the mean clustering coefficient of the graph.

    Measures how interconnected a node's neighbors are, on average
    (indicates the tendency of the network to form tightly-knit groups or
    "triangles").

    By default, edge weights are ignored. When weights are considered, the
    absolute value is taken. This is useful for correlation networks because
    their weights can be negative.

    This function uses igraph built-in function for average clustering
    calculation: "transitivity_avglocal_undirected" where mode ="zero" is
    selected. Using mode = "zero" tells to include isolated nodes in the
    calculation, adding 0 to the average sum. The alternative, mode = "nan",
    exclude the isolated nodes, resulting in an higher average. "nan" can
    hide the fact that most of the nodes are isolated, so in this case
    "zero" is more representative.

    Args:
        graph (ig.Graph): Input graph
        weight_enabled (bool): True if edge weights should be used. Default
            is False

    Returns:
        float: Average clustering coefficient
    """
    if number_of_nodes(graph) == 0:
        return 0.0

    if weight_enabled:
        if "weight" not in graph.es.attributes():
            raise ValueError(
                "weight_enabled=True requires an edge attribute named 'weight'"
            )

        # Use weighted clustering if weights are provided
        weights = [abs(weight) for weight in graph.es["weight"]]

        avr_clustering = graph.transitivity_avglocal_undirected(
            weights=weights, mode="zero"
        )
    else:
        avr_clustering = graph.transitivity_avglocal_undirected(mode="zero")

    return avr_clustering


def number_of_connected_components(graph):
    """Return the number of connected components.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        int: Number of connected components
    """
    components = graph.connected_components()
    return len(components)


def largest_connected_component(graph):
    """Return the largest connected component as a new graph.

    Return None if the graph has no nodes.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        ig.Graph: Largest connected component as a subgraph
    """
    if number_of_nodes(graph) == 0:
        return None

    components = graph.connected_components()

    largest_comp = max(components, key=len)

    return graph.subgraph(largest_comp)


def giant_component_fraction(graph):
    """Return the fraction of nodes in the largest connected component.

    Args:
        graph (ig.Graph): Input graph

    Returns:
        float: Fraction of nodes in the largest connected component
    """
    n_nodes = number_of_nodes(graph)
    if n_nodes == 0:
        return 0.0

    largest_component = largest_connected_component(graph)
    return number_of_nodes(largest_component) / n_nodes


def community_louvain(graph, weight_enabled=False):
    """Detect communities by multilevel modularity optimization (Louvain).

    Louvain groups nodes to increase network modularity. Edge weights are
    ignored by default. If weight_enabled is true, the absolute value of the
    weight edge attribute is used; so correlations with opposite signs but
    equal magnitude contribute equally to the partition.

    Args:
        graph (ig.Graph): Network whose nodes will be partitioned.
        weight_enabled (bool): Whether to use absolute edge weights.
            Defaults to False.

    Returns:
        ig.VertexClustering or None: Detected partition, or None when the
            graph has no edges.
    """
    if number_of_edges(graph) == 0:
        return None

    if weight_enabled:
        edge_weights = [abs(w) for w in graph.es["weight"]]
        return graph.community_multilevel(weights=edge_weights)
    else:
        return graph.community_multilevel()


def community_leiden(graph, weight_enabled=False):
    """Detect communities through Leiden modularity optimization.

    Leiden refines a modularity partition to avoid poorly connected groups
    that can occur with Louvain. Edge weights are ignored by default. If
    weight_enabled is true, absolute edge weights are used, so the sign of a
    correlation does not affect community detection.

    Args:
        graph (ig.Graph): Network whose nodes will be partitioned.
        weight_enabled (bool): Whether to use absolute edge weights.
            Defaults to False.

    Returns:
        ig.VertexClustering or None: Detected partition, or None when the
            graph has no edges.
    """
    if number_of_edges(graph) == 0:
        return None

    if weight_enabled:
        edge_weights = [abs(w) for w in graph.es["weight"]]
        return leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
            weights=edge_weights,
        )

    else:
        return leidenalg.find_partition(
            graph,
            leidenalg.ModularityVertexPartition,
        )


def community_infomap(graph, weight_enabled=False):
    """Detect communities by minimizing Infomap's random-walk description.

    Infomap groups nodes between which a random walker tends to remain. Edge
    weights are ignored by default. If weight_enabled is true, absolute edge
    weights determine random-walk flow, so correlation signs are ignored.

    Args:
        graph (ig.Graph): Network whose nodes will be partitioned.
        weight_enabled (bool): Whether to use absolute edge weights.
            Defaults to False.

    Returns:
        ig.VertexClustering or None: Detected partition, or None when the
            graph has no edges.
    """
    if number_of_edges(graph) == 0:
        return None

    if weight_enabled:
        edge_weights = [abs(w) for w in graph.es["weight"]]
        return graph.community_infomap(edge_weights=edge_weights)
    else:
        return graph.community_infomap()


def community_walktrap(graph, weight_enabled=False):
    """Detect communities from short random walks using Walktrap.

    Walktrap builds a hierarchy from the idea that short random walks tend
    to stay inside the same community, then returns its selected partition.
    Edge weights are ignored by default; when enabled, their absolute values
    are used and correlation signs therefore do not affect the partition.

    Args:
        graph (ig.Graph): Network whose nodes will be partitioned.
        weight_enabled (bool): Whether to use absolute edge weights.
            Defaults to False.

    Returns:
        ig.VertexClustering or None: Detected partition, or None when the
            graph has no edges.
    """
    if number_of_edges(graph) == 0:
        return None

    if weight_enabled:
        edge_weights = [abs(w) for w in graph.es["weight"]]
        return graph.community_walktrap(weights=edge_weights).as_clustering()
    else:
        return graph.community_walktrap().as_clustering()


def find_communities(graph, weight_enabled=False, algorithm="louvain"):
    """Dispatch community detection to one of the supported algorithms.

    Supported algorithms are Louvain, Leiden, Infomap, and Walktrap. Edge
    weights are ignored by default. When enabled, each algorithm uses the
    absolute value of the weight edge attribute, discarding correlation
    signs. A graph without edges has no detectable partition and returns
    None.

    Args:
        graph (ig.Graph): Input graph
        weight_enabled (bool): Whether to use absolute edge weights.
            Defaults to False.
        algorithm (str): One of "louvain", "leiden", "infomap", or
            "walktrap". Defaults to "louvain".

    Returns:
        ig.VertexClustering or None: Detected partition, or None when the
            graph has no edges.

    Raises:
        `ValueError`: If algorithm is not supported.
    """
    if number_of_edges(graph) == 0:
        return None

    algorithm = algorithm.lower()
    algorithms = {
        "louvain": community_louvain,
        "leiden": community_leiden,
        "infomap": community_infomap,
        "walktrap": community_walktrap,
    }

    if algorithm in algorithms:
        return algorithms[algorithm](graph, weight_enabled=weight_enabled)
    else:
        raise ValueError(
            f"Unknown community detection algorithm: {algorithm}."
            f"Please select one of the following: {list(algorithms.keys())}"
        )


def modularity(
    graph, communities=None, weight_enabled=False, algorithm="louvain"
):
    """Return the modularity of a community partition.

    If no partition is supplied, communities are detected automatically.
    Modularity is defined as zero for graphs without edges.

    Args:
        graph (ig.Graph): Input graph
        communities (ig.VertexClustering): List of communities as sets of
            nodes
        weight_enabled (bool): True if edge weights should be used, False
            otherwise
        algorithm (str): Algorithm for automatic community detection if
            communities=None. Choose between: "louvain", "leiden",
            "infomap", "walktrap".

    Returns:
        float: Modularity value
    """
    if number_of_edges(graph) == 0:
        return 0.0
    # If a community is not in the arguments, compute it
    if communities is None:
        communities = find_communities(
            graph, weight_enabled=weight_enabled, algorithm=algorithm
        )

    # Calculate modularity using igraph
    if weight_enabled:
        edge_weights = [abs(w) for w in graph.es["weight"]]
        return graph.modularity(communities.membership, weights=edge_weights)
    else:
        return graph.modularity(communities.membership)


def compute_network_metrics(
    graph, communities=None, weight_enabled=False, algorithm=None
):
    """Compute the main metrics of a single network.

    Args:
        graph (ig.Graph): Input graph
        communities (ig.VertexClustering, optional): Precomputed community
            partition. If omitted, it is detected using algorithm.
        weight_enabled (bool): True if edge weights should be used, False
            otherwise
        algorithm (str, optional): Algorithm for automatic community
            detection. Choose between "louvain", "leiden", "infomap", and
            "walktrap". Defaults to "louvain".

    Returns:
        dict: Dictionary of network metrics
    """
    if algorithm is None:
        algorithm = "louvain"

    if communities is None:
        communities = find_communities(
            graph,
            algorithm=algorithm,
            weight_enabled=weight_enabled,
        )

    n_nodes = number_of_nodes(graph)
    n_communities = len(communities) if communities is not None else 0
    largest_community_fraction = (
        max(len(community) for community in communities) / n_nodes
        if n_communities > 0 and n_nodes > 0
        else 0.0
    )

    metrics = {
        "nodes": n_nodes,
        "edges": number_of_edges(graph),
        "average_degree": average_degree(graph),
        "density": network_density(graph),
        "connected_components": number_of_connected_components(graph),
        "giant_component_fraction": giant_component_fraction(graph),
        "average_clustering": average_clustering(
            graph, weight_enabled=weight_enabled
        ),
        "communities": n_communities,
        "largest_community_fraction": largest_community_fraction,
        "modularity": modularity(
            graph,
            communities=communities,
            weight_enabled=weight_enabled,
            algorithm=algorithm,
        ),
    }
    return metrics
