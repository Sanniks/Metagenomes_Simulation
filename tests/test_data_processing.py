import pytest

import pandas as pd
import pandas._testing as pdt

import numpy as np

from metagenomes_simulation.data_processing import (
    group_otu_by_subject,
    absolute_to_relative_abundance,
    generate_gaussian_noise,
    clean_otus_df,
)


# TEST group_otu_by_subject
def test_group_otu_by_subject_correct_groups():
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_divided = group_otu_by_subject(otu_df, meta_df)

    otu_divided_0_expected = pd.DataFrame(
        {
            "OTU1": [10, 20],
            "OTU2": [0, 5],
            "OTU3": [2, 1],
        },
        index=["69-001-1010", "69-001-1011"],
    )

    otu_divided_1_expected = pd.DataFrame(
        {
            "OTU1": [0],
            "OTU2": [3],
            "OTU3": [4],
        },
        index=["69-074-6012"],
    )

    pdt.assert_frame_equal(otu_divided["69-001"], otu_divided_0_expected)
    pdt.assert_frame_equal(otu_divided["69-074"], otu_divided_1_expected)


def test_group_otu_by_subject_single_subject():
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-001"]},
        index=["69-001-1010", "69-001-1011", "69-001-1012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-001-1012"],
    )
    otu_divided = group_otu_by_subject(otu_df, meta_df)

    pdt.assert_frame_equal(otu_divided["69-001"], otu_df)


def test_group_otu_by_subject_error_same_indexings():
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074"]},
        index=["69-001-0", "69-001-1", "69-074-2"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


def test_group_otu_by_subject_error_missing_SubjectID_in_meta():
    meta_df = pd.DataFrame(
        {"ID": ["69-001", "69-001", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


def test_group_otu_by_subject_error_different_dimensions():
    meta_df = pd.DataFrame(
        {"SubjectID": ["69-001", "69-001", "69-074", "69-074"]},
        index=["69-001-1010", "69-001-1011", "69-074-6012", "69-074-6013"],
    )

    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    with pytest.raises(ValueError):
        group_otu_by_subject(otu_df, meta_df)


# TEST absolute_to_relative_abundance


def test_absolute_to_relative_abundance_expected():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_expected = pd.DataFrame(
        {
            "OTU1": [10 / 12, 20 / 26, 0 / 3],
            "OTU2": [0 / 12, 5 / 26, 0 / 3],
            "OTU3": [2 / 12, 1 / 26, 3 / 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_rel)


def test_absolute_to_relative_abundance_sum_of_every_row_equal_to_one():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 3],
            "OTU3": [2, 1, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)
    sample_sum_rel_abundance = otu_df_rel.sum(axis=1)
    assert np.allclose(sample_sum_rel_abundance, [1, 1, 1])


def test_absolute_to_relative_abundance_empty_sample():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 0],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)
    sample_sum_rel_abundance = otu_df_rel.sum(axis=1)
    assert np.allclose(sample_sum_rel_abundance, [1, 1, 0])


def test_absolute_to_relative_abundance_keep_index_and_column_preserved():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_rel = absolute_to_relative_abundance(otu_df_raw)

    pdt.assert_index_equal(otu_df_rel.index, otu_df_raw.index)
    pdt.assert_index_equal(otu_df_rel.columns, otu_df_raw.columns)


# TEST generate_gaussian_noise


def test_generate_gaussian_noise_output_has_same_shape():
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_noise = generate_gaussian_noise(otu_df, seed=12)

    assert otu_noise.shape == otu_df.shape


def test_generate_gaussian_noise_is_reproducible_with_same_seed():
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_noise_1 = generate_gaussian_noise(otu_df, seed=12)
    otu_noise_2 = generate_gaussian_noise(otu_df, seed=12)

    pd.testing.assert_frame_equal(otu_noise_1, otu_noise_2)


def test_generate_gaussian_noise_all_positive_values():
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_noise = generate_gaussian_noise(otu_df, seed=12)

    assert (otu_noise >= 0).all().all()


def test_generate_gaussian_noise_relative_true_normalizes_rows():
    otu_df = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )

    otu_noise = generate_gaussian_noise(otu_df, mean=1, std=0, seed=42, relative=True)

    sample_sum_rel_abundance = otu_noise.sum(axis=1)

    assert np.allclose(sample_sum_rel_abundance, 1)


def test_generate_gaussian_noise_keep_index_and_column_preserved():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 5, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_noise = generate_gaussian_noise(otu_df_raw)

    pdt.assert_index_equal(otu_noise.index, otu_df_raw.index)
    pdt.assert_index_equal(otu_noise.columns, otu_df_raw.columns)


# TEST clean_otus_df


def test_clean_otus_df_remove_empty_columns():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [0, 0, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_remove_constant_columns():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU2": [4, 4, 4],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {
            "OTU1": [10, 20, 0],
            "OTU3": [2, 1, 3],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_all_constant_columns():
    otu_df_raw = pd.DataFrame(
        {
            "OTU1": [4, 4, 4],
            "OTU2": [4, 4, 4],
            "OTU3": [4, 4, 4],
        },
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = otu_df_raw.iloc[:, 0:0]
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


def test_clean_otus_df_empty_df():
    otu_df_raw = pd.DataFrame(
        {},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_expected = pd.DataFrame(
        {},
        index=["69-001-1010", "69-001-1011", "69-074-6012"],
    )
    otu_df_new = clean_otus_df(otu_df_raw)

    pdt.assert_frame_equal(otu_expected, otu_df_new)


# Se tutte le colonne solo costanti e se nessuna colonna è costante
# Se otu_df è vuoto non fallisce
