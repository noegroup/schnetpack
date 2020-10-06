import os
import torch
import numpy as np
import pytest

import schnetpack as spk
from tests.assertions import assert_dataset_equal
from numpy.testing import assert_array_almost_equal
import ase
from ase.db import connect

from tests.fixtures import *


def test_add_and_read(empty_dataset, example_data):
    """
    Test if data can be added to existing dataset.
    """
    # add data
    for ats, props in example_data:
        empty_dataset.add_system(ats, **props)

    assert len(empty_dataset) == len(example_data)
    assert os.path.exists(empty_dataset.dbpath)


def test_empty_subset_of_subset(example_subset):
    """
    Test if empty subsubset can be created.
    """
    subsubset = spk.data.create_subset(example_subset, [])
    assert len(example_subset) == 2
    assert len(subsubset) == 0


def test_merging(tmpdir, example_dataset, partition_names):
    # create merged dataset by repeating original three times
    merged_dbpath = os.path.join(str(tmpdir), "merged.db")

    parts = [example_dataset.dbpath, example_dataset.dbpath, example_dataset.dbpath]
    if partition_names is not None:
        parts = {k: v for k, v in zip(partition_names, parts)}

    merged_data = spk.data.merge_datasets(merged_dbpath, parts)

    # check merged
    assert len(merged_data) == 3 * len(example_dataset)

    partitions = merged_data.get_metadata("partitions")
    partition_meta = merged_data.get_metadata("partition_meta")

    assert len(partitions) == 3

    for p in partitions.values():
        assert len(p) == 2

    if partition_names is not None:
        assert "example1" in partitions.keys()
        assert "example2" in partitions.keys()
        assert "ex3" in partitions.keys()


def test_loader(example_loader, batch_size):
    """
    Test dataloader iteration and batch shapes.
    """
    for batch in example_loader:
        for entry in batch.values():
            assert entry.shape[0] == min(batch_size, len(example_loader.dataset))


def test_statistics_calculation(example_loader, dataset_stats, main_properties):
    """
    Test statistics calculation of dataloader.
    """
    means, stds = example_loader.get_statistics(main_properties)
    for pname in main_properties:
        assert_array_almost_equal(means[pname].numpy(), dataset_stats[0][pname])
        assert_array_almost_equal(stds[pname].numpy(), dataset_stats[1][pname])


def test_extension_check():
    """
    Test if dataset raises error if .db is missing  in dbpath.
    """
    with pytest.raises(spk.data.AtomsDataError):
        dataset = spk.data.atoms.AtomsData("test/path")


def test_get_center(h2o, o2):
    """
    Test calculation of molecular centers.
    """
    # test if centers are equal for symmetric molecule
    com = spk.data.get_center_of_mass(o2)
    cog = spk.data.get_center_of_geometry(o2)

    assert list(com.shape) == [3]
    assert list(cog.shape) == [3]
    np.testing.assert_array_almost_equal(com, cog)

    # test if centers are different for asymmetric molecule
    com = spk.data.get_center_of_mass(h2o)
    cog = spk.data.get_center_of_geometry(h2o)

    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, com, cog)


def test_concatenation(
    example_dataset, example_subset, example_concat_dataset, example_concat_dataset2
):
    """
    Test ids of concatenated datasets.
    """
    len_e = len(example_dataset)
    # test lengths
    assert len(example_concat_dataset) == len(example_subset) + len(example_dataset)
    assert len(example_subset) == 2
    assert len(example_concat_dataset2) == len(example_concat_dataset) + len(
        example_subset
    )

    for i in range(len_e):
        c, e = example_concat_dataset[i], example_dataset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev)

    for i in range(2):
        c, e = example_concat_dataset[len_e + i], example_subset[i]
        for key in c.keys():
            if key == "_idx":
                continue
            cv, ev = c[key], e[key]
            assert torch.equal(cv, ev), "key {} does not match!".format(key)


def test_save_concatenated(tmp_data_dir, example_concat_dataset):
    """
    Test if a concatenated dataset can be saved.

    """
    # save dataset two times
    tmp_dbpath1 = os.path.join(tmp_data_dir, "concat.db")
    spk.data.save_dataset(tmp_dbpath1, example_concat_dataset)
    tmp_dbpath2 = os.path.join(tmp_data_dir, "concat2.db")
    spk.data.save_dataset(tmp_dbpath2, example_concat_dataset)

    # check if paths exist
    assert os.path.exists(tmp_dbpath1)
    assert os.path.exists(tmp_dbpath2)

    # assert if saved datasets are equal
    dataset1 = spk.data.AtomsData(tmp_dbpath1)
    dataset2 = spk.data.AtomsData(tmp_dbpath2)

    assert_dataset_equal(dataset1, dataset2)


def test_qm9(qm9_path, qm9_dataset):
    """
    Test if QM9 dataset object has same behaviour as AtomsData.

    """
    atoms_data = spk.AtomsData(qm9_path)
    assert_dataset_equal(atoms_data, qm9_dataset)


def test_md17(ethanol_path, md17_dataset):
    """
    Test if MD17 dataset object has same behaviour as AtomsData.
    """
    atoms_data = spk.AtomsData(ethanol_path)
    assert_dataset_equal(atoms_data, md17_dataset)


def test_ani1(ani1_path, ani1_dataset):
    """
    Test if MD17 dataset object has same behaviour as AtomsData.
    """
    atoms_data = spk.AtomsData(ani1_path)
    assert_dataset_equal(atoms_data, ani1_dataset)


def test_automated_metadata(random_tmp_db_path, example_data, available_properties,
                            num_data):
    # create empty dataset
    dataset = spk.data.AtomsData(random_tmp_db_path)
    assert dataset.get_metadata() == dict(
        available_properties=None,
        units=None,
        spk_version=spk.__version__,
        ase_version=ase.__version__,
        atomref=None,
    )

    # add some systems
    atoms = [d[0] for d in example_data]
    properties = [d[1] for d in example_data]
    dataset.add_systems(atoms, properties)

    # test if number of atoms is correct
    assert len(dataset) == num_data

    # test if all available_properties are set correctly
    assert dataset.available_properties == available_properties
    assert dataset.get_metadata("available_properties") == available_properties
    with connect(dataset.dbpath) as conn:
        assert set(conn.metadata["available_properties"]) == available_properties

    # add a system with missing properties
    reduced_available_properties = list(available_properties)[:-1]
    reduced_datapoint = {
        k: v for k, v in properties[0].items() if k in reduced_available_properties
    }
    dataset.add_system(atoms[0], **reduced_datapoint)

    # test if available properties has changed to reduced available properties
    reduced_available_properties = set(reduced_available_properties)
    assert dataset.available_properties == reduced_available_properties
    assert dataset.get_metadata("available_properties") == reduced_available_properties
    with connect(dataset.dbpath) as conn:
        assert set(conn.metadata["available_properties"]) == \
               reduced_available_properties

    # add system with all available properties
    dataset.add_system(atoms[0], **properties[0])

    # test if available properties is still reduced
    assert dataset.available_properties == reduced_available_properties
    assert dataset.get_metadata("available_properties") == reduced_available_properties
    with connect(dataset.dbpath) as conn:
        assert set(conn.metadata["available_properties"]) == \
               reduced_available_properties


def test_database_proptection(empty_dataset, example_data):
    # lock database and test if it throws error
    empty_dataset.protected = True

    # prepare data
    atoms = [d[0] for d in example_data]
    properties = [d[1] for d in example_data]

    # try to write data
    with pytest.raises(spk.data.AtomsDataError):
        empty_dataset.add_system(atoms[0], **properties[0])
    with pytest.raises(spk.data.AtomsDataError):
        empty_dataset.add_systems(atoms, properties)
