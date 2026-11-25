from metagenomes_simulation.network import (
    create_network_from_correlation_matrix,
    compare_network_edges,
    average_clustering,
    average_degree,
    compute_network_metrics,
    giant_component_fraction,
    largest_connected_component,
    modularity,
    network_density,
    number_of_connected_components,
    number_of_edges,
    number_of_nodes,
)
import igraph as ig
import pandas as pd
import numpy as np

import pytest

# --------------------------------------------------------------------------------------------
# TEST create_network_from_correlation_matrix
# --------------------------------------------------------------------------------------------


def test_create_network_from_correlation_matrix_keeps_isolated_nodes_default():
    """This tests that isolated taxa are kept in the network by default.

    GIVEN: a correlation matrix with one correlated pair and one isolated taxon
    WHEN: I create a network with a threshold equal to 0.5
    THEN: the network contains all three taxa and only one edge
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, 0.0],
            "OTU3": [0.0, 0.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2", "OTU3"}
    assert G.ecount() == 1


def test_create_network_from_correlation_matrix_remove_isolat_node_requested():
    """This tests that isolated taxa can still be removed explicitly.

    GIVEN: a correlation matrix with one correlated pair and one isolated taxon
    WHEN: I create a network with removal of isolated nodes enabled
    THEN: the network contains only the two taxa connected by an edge
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, 0.0],
            "OTU3": [0.0, 0.0, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        remove_isolated=True,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2"}
    assert G.ecount() == 1


def test_create_network_from_correlation_matrix_keeps_nodes_without_edges():
    """This tests that a network without edges still contains all taxa.

    GIVEN: a correlation matrix without off-diagonal values above the threshold
    WHEN: I create a network without removing isolated nodes
    THEN: the network contains every taxon and no edges
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
    )

    assert set(G.vs["name"]) == {"OTU1", "OTU2", "OTU3"}
    assert G.ecount() == 0


def test_create_network_from_correlation_matrix_thresholds_absolute_weights():
    """This tests that retained edges preserve signed correlation weights.

    GIVEN: a matrix containing positive, negative,
    and below-threshold correlations
    WHEN: I create a network with a threshold equal to 0.5
    THEN: only absolute correlations above the threshold
    become signed weighted edges
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.3],
            "OTU2": [0.8, 1.0, -0.6],
            "OTU3": [0.3, -0.6, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    positive_edge = G.es[G.get_eid("OTU1", "OTU2")]["weight"]
    negative_edge = G.es[G.get_eid("OTU2", "OTU3")]["weight"]

    assert G.ecount() == 2
    assert np.isclose(positive_edge, 0.8)
    assert np.isclose(negative_edge, -0.6)


def test_create_network_from_correlation_matrix_binary_weights():
    """This tests the creation of a binary network.

    GIVEN: a matrix with positive and negative correlations
    WHEN: I create a binary network
    THEN: every retained edge has weight one
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.8, 0.0],
            "OTU2": [0.8, 1.0, -0.6],
            "OTU3": [0.0, -0.6, 1.0],
        },
        index=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
        binary_network=True,
    )

    assert G.ecount() == 2
    assert G.es[G.get_eid("OTU1", "OTU2")]["weight"] == 1
    assert G.es[G.get_eid("OTU2", "OTU3")]["weight"] == 1


def test_create_network_from_correlation_matrix_binary_with_zero_threshold_keeps_one():
    """This tests a binary network with a one threshold.
    Edge case for threshold.

    GIVEN: a matrix with zero correlations
    WHEN: I create a binary network with threshold one
    THEN: edges are not created from nothing
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=1,
        binary_network=True,
    )

    assert G.ecount() == 0


def test_create_network_from_correlation_matrix_rejects_threshold_outside_valid_range():
    """This tests invalid threshold values.

    GIVEN: a correlation matrix
    WHEN: I use a threshold outside the interval from zero to one
    THEN: the function raises an error
    """
    correlation_matrix = pd.DataFrame(
        np.eye(2),
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(ValueError):
        create_network_from_correlation_matrix(
            correlation_matrix, threshold=-0.1
        )

    with pytest.raises(ValueError):
        create_network_from_correlation_matrix(
            correlation_matrix, threshold=1.1
        )


def test_create_network_from_correlation_matrix_keeps_edge_equal_to_threshold():
    """This tests a correlation equal to the threshold.

    GIVEN: a correlation equal to the threshold
    WHEN: I create the network
    THEN: the corresponding edge is retained
    """
    correlation_matrix = pd.DataFrame(
        {
            "OTU1": [1.0, 0.5],
            "OTU2": [0.5, 1.0],
        },
        index=["OTU1", "OTU2"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        threshold=0.5,
    )

    assert G.ecount() == 1
    assert G.es[0]["weight"] == 0.5


def test_create_network_from_correlation_matrix_removes_all_isolated_nodes():
    """This tests removal when every node is isolated.

    GIVEN: a matrix without correlations between different OTUs
    WHEN: I request the removal of isolated nodes
    THEN: the result is an empty graph
    """
    correlation_matrix = pd.DataFrame(
        np.eye(3),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    G = create_network_from_correlation_matrix(
        correlation_matrix,
        remove_isolated=True,
    )

    assert G.vcount() == 0
    assert G.ecount() == 0


@pytest.mark.parametrize(
    "threshold_mode, expected_edges",
    [
        ("absolute", 2),
        ("positive", 1),
        ("negative", 1),
    ],
)
def test_create_network_from_correlation_matrix_threshold_modes(
    threshold_mode, expected_edges
):
    """This tests the different threshold modes.

    GIVEN: a correlation matrix containing positive and negative
        correlations above and below the threshold
    WHEN: I create a network using a specific threshold mode
    THEN: only the edges satisfying the threshold condition are retained
    """
    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.8, -0.7],
            [0.8, 1.0, 0.2],
            [-0.7, 0.2, 1.0],
        ],
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    network = create_network_from_correlation_matrix(
        corr_matrix,
        threshold=0.5,
        threshold_mode=threshold_mode,
    )

    assert network.ecount() == expected_edges


def test_create_network_from_correlation_matrix_invalid_threshold_mode():
    """This tests that an invalid threshold mode raises an error.

    GIVEN: a valid correlation matrix and an unknown threshold mode
    WHEN: I create the network
    THEN: the function raises a ValueError
    """
    corr_matrix = pd.DataFrame(
        [[1.0, 0.8], [0.8, 1.0]],
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(ValueError, match="Unknown threshold mode"):
        create_network_from_correlation_matrix(
            corr_matrix,
            threshold=0.5,
            threshold_mode="invalid",
        )


# --------------------------------------------------------------------------------------------
# TEST number_of_nodes
# --------------------------------------------------------------------------------------------


def test_number_of_nodes_empty_graph():
    """Test the node count for an empty graph.

    GIVEN: An empty graph.
    WHEN: Its nodes are counted.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert number_of_nodes(G) == 0


def test_number_of_nodes_multiple_nodes():
    """Test the node count for a non-empty graph.

    GIVEN: A graph with three nodes.
    WHEN: Its nodes are counted.
    THEN: The result is three.
    """
    G = ig.Graph(n=3)
    assert number_of_nodes(G) == 3


# --------------------------------------------------------------------------------------------
# TEST number_of_edges
# --------------------------------------------------------------------------------------------


def test_number_of_edges_empty_graph():
    """Test the edge count for an empty graph.

    GIVEN: An empty graph.
    WHEN: Its edges are counted.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert number_of_edges(G) == 0


def test_number_of_edges_multiple_edges():
    """Test the edge count for a graph with edges.

    GIVEN: A graph with four edges.
    WHEN: Its edges are counted.
    THEN: The result is four.
    """
    G = ig.Graph(n=4)
    G.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert number_of_edges(G) == 4


# --------------------------------------------------------------------------------------------
# TEST average_degree
# --------------------------------------------------------------------------------------------


def test_average_degree_empty_graph():
    """Test the average degree of an empty graph.

    GIVEN: An empty graph.
    WHEN: Its average degree is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert average_degree(G) == 0.0


def test_average_degree_path_graph():
    """Test the average degree of a path graph.

    GIVEN: A three-node path graph.
    WHEN: Its average degree is computed.
    THEN: The result is four thirds.
    """
    G = ig.Graph(n=3)
    G.add_edges([(0, 1), (1, 2)])
    assert average_degree(G) == 4 / 3


def test_average_degree_directed_graph_error():
    """Test that average degree rejects directed graphs.

    GIVEN: A directed graph.
    WHEN: Its average degree is requested.
    THEN: A value error is raised.
    """
    G = ig.Graph(n=2, edges=[(0, 1)], directed=True)

    with pytest.raises(ValueError):
        average_degree(G)


# --------------------------------------------------------------------------------------------
# TEST network_density
# --------------------------------------------------------------------------------------------


def test_network_density_empty_graph():
    """Test the density of an empty graph.

    GIVEN: An empty graph.
    WHEN: Its density is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert network_density(G) == 0.0


def test_network_density_complete_graph():
    """Test the density of a complete graph.

    GIVEN: A complete three-node graph.
    WHEN: Its density is computed.
    THEN: The result is one.
    """
    G = ig.Graph(n=3)
    G.add_edges([(0, 1), (0, 2), (1, 2)])
    assert network_density(G) == 1.0


def test_network_density_partial_graph():
    """Test the density of a partial graph.

    GIVEN: A nine-node graph with three edges.
    WHEN: Its density is computed.
    THEN: The result is 3/36, where 36 is the max
    combination for nine nodes.
    """
    G = ig.Graph(n=9)
    G.add_edges([(0, 1), (0, 2), (1, 2)])

    assert network_density(G) == pytest.approx(3 / 36)


# --------------------------------------------------------------------------------------------
# TEST average_clustering
# --------------------------------------------------------------------------------------------


def test_average_clustering_empty_graph():
    """Test clustering for an empty graph.

    GIVEN: An empty graph.
    WHEN: Its average clustering is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert average_clustering(G) == 0.0


def test_average_clustering_path_graph():
    """Test clustering for a path graph.

    GIVEN: A three-node path graph.
    WHEN: Its average clustering is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=3)
    G.add_edges([(0, 1), (1, 2)])
    assert average_clustering(G) == 0.0


def test_average_clustering_triangle_graph():
    """Test clustering for a triangle graph.

    GIVEN: A three-node triangle.
    WHEN: Its average clustering is computed.
    THEN: The result is one.
    """
    G = ig.Graph(n=3)
    G.add_edges([(0, 1), (1, 2), (2, 0)])
    assert average_clustering(G) == 1.0


def test_average_clustering_weighted_triangle():
    """Test weighted clustering on a triangle.

    GIVEN: A triangle with positive and negative edge weights.
    WHEN: Its weighted average clustering is computed.
    THEN: Absolute weights are used and the result is one.
    """
    G = ig.Graph(n=3, edges=[(0, 1), (1, 2), (2, 0)])
    G.es["weight"] = [1.0, -1.0, 1.0]

    assert average_clustering(G, weight_enabled=True) == pytest.approx(1.0)


def test_average_clustering_weighted_without_weights_error():
    """Test weighted clustering without a weight attribute.

    GIVEN: A non-empty graph without edge weights.
    WHEN: Weighted average clustering is requested.
    THEN: A value error is raised.
    """
    G = ig.Graph(n=2, edges=[(0, 1)])

    with pytest.raises(ValueError):
        average_clustering(G, weight_enabled=True)


# --------------------------------------------------------------------------------------------
# TEST number_of_connected_components
# --------------------------------------------------------------------------------------------


def test_number_of_connected_components_empty_graph():
    """Test component counting for an empty graph.

    GIVEN: An empty graph.
    WHEN: Its connected components are counted.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert number_of_connected_components(G) == 0


def test_number_of_connected_components_disconnected_graph():
    """Test component counting for a disconnected graph.

    GIVEN: A graph with two disconnected pairs.
    WHEN: Its connected components are counted.
    THEN: The result is two.
    """
    G = ig.Graph(n=4)
    G.add_edges([(0, 1), (2, 3)])
    assert number_of_connected_components(G) == 2


# --------------------------------------------------------------------------------------------
# TEST largest_connected_component
# --------------------------------------------------------------------------------------------


def test_largest_connected_component_empty_graph():
    """Test the largest component of an empty graph.

    GIVEN: An empty graph.
    WHEN: Its largest component is requested.
    THEN: No component is returned.
    """
    G = ig.Graph(n=0)
    assert largest_connected_component(G) is None


def test_largest_connected_component_disconnected_graph():
    """Test the largest component of a disconnected graph.

    GIVEN: A graph with components of different sizes.
    WHEN: Its largest component is requested.
    THEN: The three-node component is returned.
    """
    G = ig.Graph(n=5)
    G.add_edges([(0, 1), (2, 3), (4, 0)])
    largest_component = largest_connected_component(G)

    assert number_of_nodes(largest_component) == 3
    assert number_of_edges(largest_component) == 2


# --------------------------------------------------------------------------------------------
# TEST giant_component_fraction
# --------------------------------------------------------------------------------------------


def test_giant_component_fraction_empty_graph():
    """Test the giant-component fraction of an empty graph.

    GIVEN: An empty graph.
    WHEN: Its giant-component fraction is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    assert giant_component_fraction(G) == 0.0


def test_giant_component_fraction_disconnected_graph():
    """Test the giant-component fraction of a disconnected graph.

    GIVEN: A graph whose largest component has three of five nodes.
    WHEN: Its giant-component fraction is computed.
    THEN: The result is three fifths.
    """
    G = ig.Graph(n=5)
    G.add_edges([(0, 1), (2, 3), (4, 0)])
    assert giant_component_fraction(G) == 3.0 / 5.0


# --------------------------------------------------------------------------------------------
# TEST modularity
# --------------------------------------------------------------------------------------------


def test_modularity_empty_graph():
    """Test modularity for an empty graph.

    GIVEN: An empty graph and no communities.
    WHEN: Its modularity is computed.
    THEN: The result is zero.
    """
    G = ig.Graph(n=0)
    communities = []
    assert modularity(G, communities) == 0.0


def test_modularity_without_communities():
    """Test modularity without predefined communities.

    GIVEN: A graph without a supplied partition.
    WHEN: Its modularity is computed.
    THEN: A valid modularity value is returned.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    G.vs["name"] = ["1", "2", "3", "4", "5", "6"]
    mod = modularity(G)
    assert isinstance(mod, float)
    assert -1.0 <= mod <= 1.0


def test_modularity_can_use_or_ignore_weights():
    """Test weighted and unweighted modularity.

    GIVEN: A weighted graph and a fixed partition.
    WHEN: Modularity is computed with and without weights.
    THEN: The values differ and unweighted is the default.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    G.vs["name"] = ["1", "2", "3", "4", "5", "6"]
    G.es["weight"] = [1, 0.1, 0.1, 0.1, 1, 1]
    communities = ig.VertexClustering(G, membership=[0, 0, 0, 1, 1, 1])

    weighted_modularity = modularity(
        G, communities=communities, weight_enabled=True
    )
    unweighted_modularity = modularity(
        G, communities=communities, weight_enabled=False
    )
    default_modularity = modularity(G, communities=communities)

    assert weighted_modularity != pytest.approx(unweighted_modularity)
    assert default_modularity == pytest.approx(unweighted_modularity)


# --------------------------------------------------------------------------------------------
# TEST compute_network_metrics
# --------------------------------------------------------------------------------------------


def test_compute_network_metrics_detects_communities_by_default():
    """Test automatic community detection with the default algorithm.

    GIVEN: A graph with two disconnected triangles and no partition.
    WHEN: Its network metrics are computed.
    THEN: Louvain detects the two communities automatically.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

    metrics = compute_network_metrics(G)

    assert metrics["communities"] == 2
    assert metrics["largest_community_fraction"] == pytest.approx(0.5)


def test_compute_network_metrics_expected():
    """Test the complete network-metrics summary.

    GIVEN: A graph and its community partition.
    WHEN: All network metrics are computed.
    THEN: The expected metrics and values are returned.
    """
    G = ig.Graph(n=4)
    G.add_edges([(0, 1), (1, 2), (2, 3), (0, 3)])
    communities = ig.VertexClustering(G, membership=[0, 0, 1, 1])

    metrics = compute_network_metrics(G, communities=communities)

    assert set(metrics) == {
        "nodes",
        "edges",
        "average_degree",
        "density",
        "connected_components",
        "giant_component_fraction",
        "average_clustering",
        "communities",
        "largest_community_fraction",
        "modularity",
    }
    assert metrics["nodes"] == 4
    assert metrics["edges"] == 4
    assert metrics["communities"] == 2
    assert metrics["largest_community_fraction"] == pytest.approx(0.5)
    assert metrics["modularity"] == pytest.approx(
        modularity(G, communities=communities, weight_enabled=False)
    )


# --------------------------------------------------------------------------------------------
# TEST compare_network_edges
# --------------------------------------------------------------------------------------------


def test_compare_network_edges_expected():
    """This tests the calculation of network comparison metrics.

    GIVEN: two networks with shared and different edges
    WHEN: I compare their edges
    THEN: the expected comparison metrics are returned
    """
    network = ig.Graph(
        n=4,
        edges=[(0, 1), (0, 2), (1, 2)],
    )
    ground_truth_network = ig.Graph(
        n=4,
        edges=[(0, 1), (0, 2), (2, 3)],
    )

    network.vs["name"] = ["OTU1", "OTU2", "OTU3", "OTU4"]
    ground_truth_network.vs["name"] = network.vs["name"]

    results = compare_network_edges(network, ground_truth_network)

    expected = {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
        "jaccard": 1 / 2,
    }

    assert results == pytest.approx(expected)


def test_compare_network_edges_identical_networks():
    """This tests the comparison of identical networks.

    GIVEN: two networks containing the same edges
    WHEN: I compare their edges
    THEN: all edges are true positives and all scores are one
    """
    network = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
    network.vs["name"] = ["OTU1", "OTU2", "OTU3"]

    results = compare_network_edges(network, network.copy())

    expected = {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "precision": 1,
        "recall": 1,
        "f1": 1,
        "jaccard": 1,
    }

    assert results == pytest.approx(expected)


@pytest.mark.parametrize(
    "observed_edges, ground_truth_edges, expected_fp, expected_fn",
    [
        ([], [], 0, 0),
        ([(0, 1)], [], 1, 0),
        ([], [(0, 1)], 0, 1),
    ],
)
def test_compare_network_edges_empty_networks(
    observed_edges, ground_truth_edges, expected_fp, expected_fn
):
    """This tests network comparison when edges are absent.

    GIVEN: two networks with at least one network containing no edges
    WHEN: I compare their edges
    THEN: the expected edge counts are returned and all scores are zero
    """
    network = ig.Graph(n=3, edges=observed_edges)
    ground_truth_network = ig.Graph(n=3, edges=ground_truth_edges)

    network.vs["name"] = ["OTU1", "OTU2", "OTU3"]
    ground_truth_network.vs["name"] = network.vs["name"]

    results = compare_network_edges(network, ground_truth_network)

    expected = {
        "tp": 0,
        "fp": expected_fp,
        "fn": expected_fn,
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "jaccard": 0,
    }

    assert results == pytest.approx(expected)


def test_compare_network_edges_ignores_weights():
    """This tests that edge comparison ignores weights and signs.

    GIVEN: two networks with identical edges but different weights
    WHEN: I compare their edges
    THEN: the networks are considered identical
    """
    network = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
    network.vs["name"] = ["OTU1", "OTU2", "OTU3"]
    network.es["weight"] = [0.8, -0.6]

    ground_truth_network = network.copy()
    ground_truth_network.es["weight"] = [-0.9, 0.3]

    results = compare_network_edges(network, ground_truth_network)

    assert results["tp"] == 2
    assert results["fp"] == 0
    assert results["fn"] == 0
    assert results["f1"] == pytest.approx(1)
    assert results["jaccard"] == pytest.approx(1)


def test_compare_network_edges_error_network_not_igraph():
    """This tests that an invalid observed network raises an error.

    GIVEN: an observed network that is not an igraph Graph
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = 4
    ground_truth_network = ig.Graph()
    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_ground_truth_network_not_igraph():
    """This tests that an invalid ground-truth network raises an error.

    GIVEN: a ground-truth network that is not an igraph Graph
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph()
    ground_truth_network = 4
    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_directed_network():
    """This tests that directed networks raise an error.

    GIVEN: a directed network and an undirected network
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3, directed=True)
    ground_truth_network = ig.Graph(n=3)

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_self_loop():
    """This tests that networks containing self-loops raise an error.

    GIVEN: a network containing a self-loop
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3, edges=[(0, 0)])
    ground_truth_network = ig.Graph(n=3)

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_different_node_count():
    """This tests that networks with different node counts raise an error.

    GIVEN: two networks containing different numbers of nodes
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3)
    ground_truth_network = ig.Graph(n=4)

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_missing_node_names():
    """This tests that networks without node names raise an error.

    GIVEN: a network without node names
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3)
    ground_truth_network = ig.Graph(n=3)
    ground_truth_network.vs["name"] = ["OTU1", "OTU2", "OTU3"]

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_different_node_names():
    """This tests that networks containing different taxa raise an error.

    GIVEN: two networks containing different node names
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3)
    ground_truth_network = ig.Graph(n=3)

    network.vs["name"] = ["OTU1", "OTU2", "OTU3"]
    ground_truth_network.vs["name"] = ["OTU1", "OTU2", "OTU4"]

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)


def test_compare_network_edges_error_different_node_order():
    """This tests that different node orderings raise an error.

    GIVEN: two networks containing the same taxa in different orders
    WHEN: I compare the network edges
    THEN: the function raises an error
    """
    network = ig.Graph(n=3)
    ground_truth_network = ig.Graph(n=3)

    network.vs["name"] = ["OTU1", "OTU2", "OTU3"]
    ground_truth_network.vs["name"] = ["OTU2", "OTU1", "OTU3"]

    with pytest.raises(ValueError):
        compare_network_edges(network, ground_truth_network)
