import pytest
import igraph as ig

from metagenomes_simulation.communities import (
    community_infomap,
    community_leiden,
    community_louvain,
    community_walktrap,
    find_communities,
)


# --------------------------------------------------------------------------------------------
# TEST community detection algorithms
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "community_function",
    [
        community_louvain,
        community_leiden,
        community_infomap,
        community_walktrap,
    ],
)
def test_community_algorithms_find_two_disconnected_communities(
    community_function,
):
    """Test community detection on disconnected triangles.

    GIVEN: Two disconnected triangles.
    WHEN: A community algorithm is applied.
    THEN: Each triangle forms one community.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

    communities = community_function(G)
    detected = []
    for community in communities:
        detected.append(sorted(community))

    expected = [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert sorted(detected) == sorted(expected)


@pytest.mark.parametrize(
    "community_function",
    [
        community_louvain,
        community_leiden,
        community_infomap,
        community_walktrap,
    ],
)
def test_community_algorithms_empty_graph(community_function):
    """Test community detection on an empty graph.

    GIVEN: An empty graph.
    WHEN: A community algorithm is applied.
    THEN: No communities are returned.
    """
    assert community_function(ig.Graph(n=0)) is None


def test_community_louvain_weighted():
    """Test weighted Louvain community detection.

    GIVEN: Two weighted disconnected triangles.
    WHEN: Weighted Louvain is applied.
    THEN: Each triangle forms one community.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    G.es["weight"] = [1, 1, 1, 1, 1, 1]

    communities = community_louvain(G, weight_enabled=True)
    detected = []
    for community in communities:
        detected.append(sorted(community))

    expected = [
        [0, 1, 2],
        [3, 4, 5],
    ]

    assert sorted(detected) == sorted(expected)


# --------------------------------------------------------------------------------------------
# TEST find_communities
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "community_function",
    [
        "louvain",
        "leiden",
        "infomap",
        "walktrap",
    ],
)
def test_find_communities_valid_algorithm_with_weights(community_function):
    """Test weighted community detection with a valid algorithm.

    GIVEN: A weighted graph and an algorithm.
    WHEN: Communities are requested.
    THEN: A vertex clustering is returned.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    G.es["weight"] = [1, 2, 3, 4, 5, 6]

    communities = find_communities(
        G, algorithm=community_function, weight_enabled=True
    )

    assert isinstance(communities, ig.VertexClustering)


def test_find_communities_invalid_algorithm():
    """Test community detection with an invalid algorithm.

    GIVEN: A graph and an unknown algorithm name.
    WHEN: Communities are requested.
    THEN: A value error is raised.
    """
    G = ig.Graph(n=6)
    G.add_edges([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

    with pytest.raises(ValueError):
        find_communities(G, algorithm="invalid_algorithm")


def test_find_communities_graph_without_edges():
    """Test community detection on a graph without edges.

    GIVEN: A graph containing only isolated nodes.
    WHEN: Communities are requested.
    THEN: No partition is returned.
    """
    assert find_communities(ig.Graph(n=3)) is None
