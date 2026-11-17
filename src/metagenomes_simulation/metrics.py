import numpy as np
import igraph as ig

from metagenomes_simulation.communities import find_communities

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
# ---------------------------------------------------------------------------


def compare_correlation_matrices(observed_corr, ground_truth_corr):
    """Compare the upper triangles of two correlation matrices. It measures the
    Bias, the Mean Absolute Error and the Root Mean Squared Error between
    the two matrices.

    The diagonal is excluded from the comparison since it constist of all 1.

    Args:
        observed_corr (pandas.DataFrame): Observed correlation matrix.
        ground_truth_corr (pandas.DataFrame): Ground-truth correlation
            matrix.

    Returns:
        dict: Bias, MAE, RMSE, and Pearson correlation coefficient between
            the corresponding off-diagonal correlations.

    Raises:
        `ValueError`: If the input matrices do not have the same dimensions.
    """
    if observed_corr.shape != ground_truth_corr.shape:
        raise ValueError("Input matrices must have the same dimension")
    if set(observed_corr.index) != set(ground_truth_corr.index):
        raise ValueError("Input matrices must contain the same taxa")

    # If the OTUs are not aligned, they become aligned before the calculations
    ground_truth_corr = ground_truth_corr.loc[
        observed_corr.index, observed_corr.columns
    ]

    indices = np.triu_indices_from(observed_corr.to_numpy(), k=1)

    observed_corr_np = observed_corr.to_numpy()[indices]
    ground_truth_corr_np = ground_truth_corr.to_numpy()[indices]

    errors = observed_corr_np - ground_truth_corr_np

    results = {
        "bias": np.mean(errors),
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors**2)),
    }
    return results


def compare_network_edges(observed_corr, ground_truth_corr, threshold=0.3):
    """Compare the edges of observed and ground-truth correlation networks.

    Args:
        observed_corr (pd.DataFrame): Estimated taxon correlation matrix.
        ground_truth_corr (pd.DataFrame): Ground-truth taxon correlation
            matrix.
        threshold (float): Absolute threshold applied to the observed_corr
            matrix. Defaults to 0.3.

    Returns:
        dict: True positives ("tp"), false positives ("fp"), false negatives
            ("fn"), precision, recall, F1 score, and edge Jaccard
            similarity. Precision or recall is "numpy.nan" when its
            denominator is zero. F1 and Jaccard are one when both networks
            contain no edges.

    Raises:
        `ValueError`: If the two matrices have different dimensions.
    """
    if observed_corr.shape != ground_truth_corr.shape:
        raise ValueError("Input matrices must have the same dimension")
    if set(observed_corr.index) != set(ground_truth_corr.index):
        raise ValueError("Input matrices must contain the same taxa")

    # If the OTUs are not aligned, they become
    # aligned before the calculations
    ground_truth_corr = ground_truth_corr.loc[
        observed_corr.index, observed_corr.columns
    ]

    # Take only the upper triangle of the matrix
    # and only values above the trashold
    indices = np.triu_indices_from(observed_corr.to_numpy(), k=1)
    observed_corr_edges = (
        np.abs(observed_corr.to_numpy()[indices]) >= threshold
    )
    true_edges = np.abs(ground_truth_corr.to_numpy()[indices]) >= threshold

    # Count True Positive, False Positive and False Negative,
    # comparing the simulated correlation matrix
    # to the ground truth matrix
    tp = np.count_nonzero(observed_corr_edges & true_edges)
    fp = np.count_nonzero(observed_corr_edges & ~true_edges)
    fn = np.count_nonzero(~observed_corr_edges & true_edges)

    # Measure network matrices
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = np.nan

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = np.nan

    f1_denominator = 2 * tp + fp + fn

    if f1_denominator != 0:
        f1 = 2 * tp / f1_denominator
    else:
        f1 = 1.0

    union = tp + fp + fn
    if union != 0:
        jaccard = tp / union
    else:
        jaccard = 1.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }
    return results


def compare_communities(
    observed_community_memberships, ground_truth_community_memberships
):

    vi = ig.compare_communities(
        ground_truth_community_memberships,
        observed_community_memberships,
        method="vi",
        remove_none=True,
    )
    ari = ig.compare_communities(
        ground_truth_community_memberships,
        observed_community_memberships,
        method="adjusted_rand",
        remove_none=True,
    )
    nmi = ig.compare_communities(
        ground_truth_community_memberships,
        observed_community_memberships,
        method="nmi",
        remove_none=True,
    )

    results = {"vi": vi, "ari": ari, "nmi": nmi}
    return results
