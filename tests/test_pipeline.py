from metagenomes_simulation.pipeline import simulate_metagenomes_pipeline
import numpy as np
import pytest

parameters = {
    # Production parameters
    "community_blocks": [5, 5],
    "within_density": 0.5,
    "correlation_range": (0.5, 0.9),
    "n_sample": 50,
    "mean_abundance": 1.0,
    "var_abundance": 0.1,
    "sequencing_depth": 1000,
    # Analysis parameters
    "thresholds": np.array([0.1, 0.3, 0.5]),
    "threshold_mode": "positive",
    "binary_network": True,
    "community_algorithms": ("louvain", "leiden", "infomap", "walktrap"),
    # Computational parameters
    "num_processes": 1,
    "random_seed": 1234,
}

# Execute pipeline only once
@pytest.fixture(scope="module")
def pipeline_result():
    return simulate_metagenomes_pipeline(**parameters)

# --------------------------------------------------------------------------------------------
# TEST simulate_metagenomes_pipeline
# --------------------------------------------------------------------------------------------


def test_simulate_metagenomes_pipeline_expected_keys(pipeline_result):
    """This tests the structure of the pipeline results.

    GIVEN: a valid set of simulation parameters
    WHEN: I execute the simulation pipeline
    THEN: all expected results are returned
    """
    expected_keys = {
        "correlation_ground_truth",
        "true_membership",
        "absolute_abundances_ground_truth",
        "otus_absolute",
        "otus_relative",
        "otus_clr",
        "correlations",
        "error_values",
        "ground_truth_network",
        "networks",
        "network_metrics",
        "communities",
        "community_metrics",
    }

    assert set(pipeline_result.keys()) == expected_keys
    
def test_simulate_metagenomes_pipeline_expected_dimensions(pipeline_result):
    """This tests the dimensions of the simulated data.

    GIVEN: a specified number of taxa and samples
    WHEN: I execute the simulation pipeline
    THEN: the simulated matrices have the expected dimensions
    """
    n_taxa = sum(parameters["community_blocks"])
    n_sample = parameters["n_sample"]

    assert pipeline_result["correlation_ground_truth"].shape == (
        n_taxa, n_taxa
    )
    assert pipeline_result["true_membership"].shape == (n_taxa,)
    assert pipeline_result["absolute_abundances_ground_truth"].shape == (
        n_sample, n_taxa
    )
    assert pipeline_result["otus_absolute"].shape == (
        n_sample, n_taxa
    )
    assert pipeline_result["otus_relative"].shape == (
        n_sample, n_taxa
    )
    
def test_simulate_metagenomes_pipeline_expected_normalizations(
    pipeline_result,
):
    """This tests that all abundance representations are analyzed.

    GIVEN: a valid set of simulation parameters
    WHEN: I execute the simulation pipeline
    THEN: correlations are returned for all abundance representations
    """
    expected = {"Absolute", "Relative", "CLR"}

    assert set(pipeline_result["correlations"].keys()) == expected
    
def test_simulate_metagenomes_pipeline_expected_thresholds(pipeline_result):
    """This tests that all requested thresholds are evaluated.

    GIVEN: a set of correlation thresholds
    WHEN: I execute the simulation pipeline
    THEN: networks and comparison metrics are returned for every threshold
    """
    expected = set(parameters["thresholds"])

    assert set(pipeline_result["networks"]) == expected
    assert set(pipeline_result["network_metrics"]) == expected
    assert set(pipeline_result["communities"]) == expected
    assert set(pipeline_result["community_metrics"]) == expected
    
def test_simulate_metagenomes_pipeline_network_node_order(pipeline_result):
    """This tests that reconstructed networks preserve the node ordering.

    GIVEN: a ground-truth network and reconstructed networks
    WHEN: I execute the simulation pipeline
    THEN: every network contains the same ordered taxa
        as the ground-truth network
    """
    expected = pipeline_result["ground_truth_network"].vs["name"]

    for normalizations in pipeline_result["networks"].values():
        for methods in normalizations.values():
            for network in methods.values():
                assert network.vs["name"] == expected
                
def test_simulate_metagenomes_pipeline_expected_algorithms(pipeline_result):
    """This tests that all requested community algorithms are evaluated.

    GIVEN: a set of community detection algorithms
    WHEN: I execute the simulation pipeline
    THEN: community comparison metrics are returned for every algorithm
    """
    expected = set(parameters["community_algorithms"])

    for normalizations in pipeline_result["community_metrics"].values():
        for methods in normalizations.values():
            for algorithms in methods.values():
                assert set(algorithms.keys()) == expected
                
                
def test_simulate_metagenomes_pipeline_valid_network_metrics(pipeline_result):
    """This tests the validity of network reconstruction metrics.

    GIVEN: reconstructed networks and a ground-truth network
    WHEN: I execute the simulation pipeline
    THEN: all edge counts are non-negative and all scores
        are between zero and one
    """
    for normalizations in pipeline_result["network_metrics"].values():
        for methods in normalizations.values():
            for metrics in methods.values():
                for key in ("tp", "fp", "fn"):
                    assert metrics[key] >= 0

                for key in ("precision", "recall", "f1", "jaccard"):
                    assert 0 <= metrics[key] <= 1