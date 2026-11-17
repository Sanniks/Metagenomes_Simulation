import pytest
import igraph as ig
import pandas as pd
import numpy as np

from metagenomes_simulation.metrics import (
    compare_network_edges,
    compare_correlation_matrices,
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
# TEST compute_network_metrics
# --------------------------------------------------------------------------------------------
def test_compare_correlation_matrices_expected_metrics_identical():

    A = pd.DataFrame(
        np.ones((3, 3)),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    B = pd.DataFrame(
        np.ones((3, 3)),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    results = compare_correlation_matrices(A, B)
    expected = {
        "bias": 0,
        "mae": 0,
        "rmse": 0,
    }
    assert expected == pytest.approx(results)


def test_compare_correlation_matrices_error_different_shape():

    A = pd.DataFrame(
        np.ones((3, 3)),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )
    B = pd.DataFrame(
        np.ones((2, 2)),
        index=["OTU1", "OTU2"],
        columns=["OTU1", "OTU2"],
    )

    with pytest.raises(ValueError):
        compare_correlation_matrices(A, B)


def test_compare_correlation_matrices_error_different_index():

    A = pd.DataFrame(
        np.ones((3, 3)),
        index=["OTU_2", "OTU_1", "OTU_3"],
        columns=["OTU_2", "OTU_1", "OTU_3"],
    )
    B = pd.DataFrame(
        np.ones((3, 3)),
        index=["OTU1", "OTU2", "OTU3"],
        columns=["OTU1", "OTU2", "OTU3"],
    )

    with pytest.raises(ValueError):
        compare_correlation_matrices(A, B)
