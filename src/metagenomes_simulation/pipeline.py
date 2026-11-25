import numpy as np

from metagenomes_simulation.generators import (
    create_sparse_block_correlation,
    simulate_absolute_abundances_ground_truth,
    simulate_sequencing,
)
from metagenomes_simulation.normalization import (
    normalize_abundance_all_methods,
)
from metagenomes_simulation.correlation import measure_correlation_all_methods
from metagenomes_simulation.network import (
    create_network_from_correlation_matrix,
    compare_network_edges,
)
from metagenomes_simulation.communities import (
    find_communities,
    compare_communities,
)


def simulate_metagenomes_pipeline(
    community_blocks,
    within_density,
    correlation_range,
    n_sample,
    mean_abundance,
    var_abundance,
    sequencing_depth,
    thresholds,
    threshold_mode,
    binary_network,
    community_algorithms,
    num_processes,
    random_seed,
):
    """Run the complete simulated metagenomics analysis pipeline.

    This function generate a block-sparse correlation matrix, then produces
    abundances with a fixed number of samples. Then it simulates the
    sequencing of the abundances with a Poisson-distributed sequencing
    depth.

    The function analyzes the sequencing counts using three representations:
    raw counts (Absolute), relative abundances, and centered log-ratio (CLR)
    transformed abundances. It estimates taxon correlations using Pearson,
    Spearman, and Kendall methods.

    From these correlation matrix, networks are built. They are compared
    with the ground truth networks, F1 and jaccard metrics are measured.
    Then it finds the communities using "louvain", "leiden", "infomap" and
    "walktrap" algorithms.

    They are compared with the true communities (used to build the ground
    truth correlation matrices), using Variation of Information (VI),
    Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

    Args:
        community_blocks (list[int]): Number of taxa in each ground-truth
            community.
        within_density (float): Density of nonzero correlations within each
            community.
        correlation_range (tuple[float, float]): Minimum and maximum
            correlation strengths used to generate the ground truth.
        n_sample (int): Number of simulated metagenomic samples.
        mean_abundance (float): Mean parameter used to generate the absolute
            abundance distribution.
        var_abundance (float): Variance parameter used to generate the
            absolute abundance distribution.
        sequencing_depth (int): Mean of the Poisson distribution used to
            generate the sequencing depth of each sample.
        thresholds (array-like or None): Correlation thresholds used to
            construct the inferred networks. If None, default thresholds are
            used.
        binary_network (bool): All edges above the threshold have weigth 1
            instead of the correlation coefficient.
        community_algorithms (tuple[str, ...]): Community detection
            algorithms to evaluate. Defaults to Louvain, Leiden, Infomap,
            and Walktrap.
        num_processes (int): Number of processes used for correlation
            estimation. Defaults to 4.
        random_seed (int or None): Random seed for reproducibility.

    Returns:
        dict: Results of the simulation, including the ground-truth
            correlations and memberships, simulated abundances, sequencing
            counts, estimated correlation matrices, reconstructed networks,
            edge reconstruction metrics, detected communities, and community
            comparison metrics.
    """

    # Create seeds for each random function
    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = []
    for rng in seed_sequence.spawn(3):
        seeds.append(int(rng.generate_state(1)[0]))

    seed_ground_truth, seed_abundances, seed_sequencing = seeds

    # Simulate sequencing
    correlation_ground_truth, true_membership = (
        create_sparse_block_correlation(
            block_sizes=community_blocks,
            within_density=within_density,
            correlation_range=correlation_range,
            random_state=seed_ground_truth,
        )
    )
    absolute_abundances_ground_truth = (
        simulate_absolute_abundances_ground_truth(
            correlation_ground_truth,
            n_sample=n_sample,
            mean_abundance=mean_abundance,
            var_abundance=var_abundance,
            random_state=seed_abundances,
        )
    )
    otus_absolute = simulate_sequencing(
        absolute_abundances_ground_truth,
        sequencing_depth=sequencing_depth,
        random_state=seed_sequencing,
    )

    # Normalize seqeuncing
    normalized = normalize_abundance_all_methods(otus_absolute)
    otus_relative = normalized["relative"]
    otus_clr = normalized["clr"]

    # Calculate correlation matrices
    absolute_corr = measure_correlation_all_methods(
        otus_absolute, num_processes=num_processes
    )
    relative_corr = measure_correlation_all_methods(
        otus_relative, num_processes=num_processes
    )
    clr_corr = measure_correlation_all_methods(
        otus_clr, num_processes=num_processes
    )

    correlations = {
        "Absolute": absolute_corr,
        "Relative": relative_corr,
        "CLR": clr_corr,
    }

    # Difference with ground-truth correlation matrix
    error_values = {}

    for normalization, methods in correlations.items():
        error_values[normalization] = {}

        for method, corr_matrix in methods.items():
            otu_names = corr_matrix.columns
            filtered_ground_truth = correlation_ground_truth.loc[
                otu_names, otu_names
            ]
            errors = corr_matrix - filtered_ground_truth

            error_matrix = errors.to_numpy()
            errors_values = error_matrix[
                np.triu_indices_from(error_matrix, k=1)
            ]

            bias = np.mean(errors_values)
            mae = np.mean(np.abs(errors_values))
            rmse = np.sqrt(np.mean(errors_values**2))

            error_values[normalization][method] = {
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
            }

    # Create ground truth network
    ground_truth_network = create_network_from_correlation_matrix(
        correlation_ground_truth,
        threshold=1e-10,
        remove_isolated=False,
        binary_network=binary_network,
        threshold_mode="absolute",
    )

    # Create networks with different thresholds
    networks = {}

    for threshold in thresholds:
        networks[threshold] = {}

        for normalization, corr in correlations.items():
            networks[threshold][normalization] = {}

            for method, corr_matrix in corr.items():
                network = create_network_from_correlation_matrix(
                    corr_matrix,
                    threshold,
                    binary_network=binary_network,
                    threshold_mode=threshold_mode,
                )

                # Add missing OTUs
                missing_otus = []

                for otu in correlation_ground_truth.columns:
                    if otu not in network.vs["name"]:
                        missing_otus.append(otu)

                network.add_vertices(missing_otus)

                # Reorder vertices to match the ground truth
                new_order = []

                for otu in correlation_ground_truth.columns:
                    index = network.vs.find(name=otu).index
                    new_order.append(index)

                network = network.permute_vertices(new_order)
                networks[threshold][normalization][method] = network

    # Find F!, Jaccard, Recall and Precision
    network_metrics = {}

    for threshold, normalizations in networks.items():
        network_metrics[threshold] = {}

        for normalization, methods in normalizations.items():
            network_metrics[threshold][normalization] = {}

            for method, network in methods.items():
                network_metrics[threshold][normalization][method] = (
                    compare_network_edges(network, ground_truth_network)
                )

    # Compare starting communities with observed ones
    communities = {}
    community_metrics = {}

    for threshold, normalizations in networks.items():
        communities[threshold] = {}
        community_metrics[threshold] = {}

        for normalization, methods in normalizations.items():
            communities[threshold][normalization] = {}
            community_metrics[threshold][normalization] = {}

            for method, network in methods.items():
                communities[threshold][normalization][method] = {}
                community_metrics[threshold][normalization][method] = {}

                for algorithm in community_algorithms:

                    # Community detection
                    partition = find_communities(
                        network,
                        weight_enabled=not binary_network,
                        algorithm=algorithm,
                    )

                    communities[threshold][normalization][method][
                        algorithm
                    ] = partition

                    # Community comparison
                    if partition is None:
                        metrics = {
                            "vi": np.nan,
                            "ari": np.nan,
                            "nmi": np.nan,
                        }
                    else:
                        metrics = compare_communities(
                            observed_community_memberships=partition.membership,
                            ground_truth_community_memberships=true_membership,
                        )

                    community_metrics[threshold][normalization][method][
                        algorithm
                    ] = metrics
    return {
        "correlation_ground_truth": correlation_ground_truth,
        "true_membership": true_membership,
        "absolute_abundances_ground_truth": absolute_abundances_ground_truth,
        "otus_absolute": otus_absolute,
        "otus_relative": otus_relative,
        "otus_clr": otus_clr,
        "correlations": correlations,
        "error_values": error_values,
        "ground_truth_network": ground_truth_network,
        "networks": networks,
        "network_metrics": network_metrics,
        "communities": communities,
        "community_metrics": community_metrics,
    }
