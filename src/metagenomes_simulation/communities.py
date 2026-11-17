import leidenalg


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
    if not graph.es:
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
    if not graph.es:
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
    if not graph.es:
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
    if not graph.es:
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
    if not graph.es:
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
