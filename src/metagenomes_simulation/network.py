import numpy as np
import pandas as pd
import igraph as ig

from metagenomes_simulation.communities import find_communities

# --------------------------------------------------------------
# Correlations network from correlation matrix using igraph
# --------------------------------------------------------------


def create_network_from_correlation_matrix(
    correlation_matrix,
    threshold=0.1,
    remove_isolated=False,
    binary_network=False,
    threshold_mode="absolute",
):
    """Create an igraph network from a correlation matrix.

    It also removes all edges of weigth lower than the threshold.

    Args:
        correlation_matrix (pd.DataFrame): Matrix containing the correlation
            coefficients between OTUs.
        threshold (float, optional): Threshold to cut the smaller edges.
            Defaults to 0.1.
        remove_isolated (bool, optional): Allows to remove any isolated
            nodes. Defaults to False.
        binary_network (bool, optional): All edges above the threshold have
            weigth 1 instead of the correlation coefficient. Defaults to
            False.

    Returns:
        ig.Graph: igraph network built from the correlation matrix.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    # Copy the input matrix as numpy array
    adjacency_np = correlation_matrix.to_numpy(copy=True)

    # Removes self-loops
    np.fill_diagonal(adjacency_np, 0)

    # Convert back to a DataFrame
    adjacency = pd.DataFrame(
        adjacency_np,
        index=correlation_matrix.index,
        columns=correlation_matrix.columns,
    )

    # If an edge is lower to the threshold, it removes it
    if threshold_mode == "absolute":
        adjacency[adjacency.abs() < threshold] = 0

    elif threshold_mode == "positive":
        adjacency[adjacency < threshold] = 0

    elif threshold_mode == "negative":
        adjacency[adjacency > -threshold] = 0

    else:
        raise ValueError("Unknown threshold mode.")

    if binary_network:
        # If an edge is higher than the threshold, it makes it 1
        adjacency[adjacency != 0] = 1
    if remove_isolated:
        # Keeps only nodes with edges
        nodes_accepted = adjacency.abs().max(axis=1) > 0
        filtered_correlation_matrix = adjacency.loc[
            nodes_accepted, nodes_accepted
        ]
    else:
        filtered_correlation_matrix = adjacency

    # If every node is removed
    if filtered_correlation_matrix.shape[0] == 0:
        return ig.Graph(n=0)

    # Create a igraph network
    graph_ig = ig.Graph.Weighted_Adjacency(
        matrix=filtered_correlation_matrix.to_numpy(copy=True),
        mode="undirected",
        attr="weight",
        loops=False,
    )

    # Assign a name to each OTU node
    graph_ig.vs["name"] = [
        str(col) for col in filtered_correlation_matrix.columns
    ]
    return graph_ig


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
    graph, communities=None, weight_enabled=False, algorithm="louvain"
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


# ---------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------


def compare_network_edges(network, ground_truth_network):
    """Compare the edges of observed and ground-truth correlation networks.

    Compare two undirected networks by considering the presence or absence
    of edges, independently of their weights and signs.

    Args:
        network (ig.Graph): Reconstructed correlation network.
        ground_truth_network (ig.Graph): Ground-truth correlation network.

    Returns:
        dict: True positives (tp), false positives (fp), false negatives
            (fn), precision, recall, F1 score, and Jaccard similarity.
            Metrics are set to zero when their denominator is zero.

    Raises:
        `ValueError`: If the inputs are not igraph graphs, are directed, are
                not simple graphs, or do not contain the same taxa in the
                same order.
    """
    if not isinstance(network, ig.Graph):
        raise ValueError("network must be an igraph Graph.")

    if not isinstance(ground_truth_network, ig.Graph):
        raise ValueError("ground_truth_network must be an igraph Graph.")

    if network.is_directed() or ground_truth_network.is_directed():
        raise ValueError("Input networks must be undirected.")

    if not network.is_simple() or not ground_truth_network.is_simple():
        raise ValueError("Input networks must be simple graphs.")

    if network.vcount() != ground_truth_network.vcount():
        raise ValueError("Input networks must have the same number of nodes.")

    if (
        "name" not in network.vs.attributes()
        or "name" not in ground_truth_network.vs.attributes()
    ):
        raise ValueError("Network nodes must have names.")

    if network.vs["name"] != ground_truth_network.vs["name"]:
        raise ValueError("Input networks must contain the same ordered taxa.")

    # Count true positives, false positives and false negatives
    tp = (network & ground_truth_network).ecount()
    fp = (network - ground_truth_network).ecount()
    fn = (ground_truth_network - network).ecount()

    # Precision
    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # Recall
    if tp + fn:
        recall = tp / (tp + fn)
    else:
        recall = 0

    # F1 score
    if 2 * tp + fp + fn:
        f1 = 2 * tp / (2 * tp + fp + fn)
    else:
        f1 = 0

    # Jaccard similarity
    if tp + fp + fn:
        jaccard = tp / (tp + fp + fn)
    else:
        jaccard = 0

    results = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }

    return results
