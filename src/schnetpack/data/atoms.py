"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import os
import warnings
import bisect

import numpy as np
import torch
import ase
from ase.db import connect
from torch.utils.data import Dataset, ConcatDataset, Subset

import schnetpack as spk
from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider, collect_atom_triples

from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "AtomsData",
    "AtomsDataSubset",
    "ConcatAtomsData",
    "AtomsDataError",
    "AtomsConverter",
    "get_center_of_mass",
    "get_center_of_geometry",
]


def get_center_of_mass(atoms):
    """
    Computes center of mass.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of mass
    """
    masses = atoms.get_masses()
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()


def get_center_of_geometry(atoms):
    """
    Computes center of geometry.

    Args:
        atoms (ase.Atoms): atoms object of molecule

    Returns:
        center of geometry
    """
    return atoms.arrays["positions"].mean(0)


class AtomsDataError(Exception):
    pass


class AtomsData(Dataset):
    """
    PyTorch dataset for atomistic data. The raw data is stored in the specified
    ASE database. Use together with schnetpack.data.AtomsLoader to feed data
    to your model.

    Args:
        dbpath (str): path to directory containing database.
        subset (list, optional): Deprecated! Do not use! Subsets are created with
            AtomsDataSubset class.
        available_properties (list, optional): Deprecated! complete set of physical
            properties that are contained in the database.
        load_only (list, optional): reduced set of properties to be loaded
        units (dict, optional): definition of units for all available properties
        environment_provider (spk.environment.BaseEnvironmentProvider): define how
            neighborhood is calculated
            (default=spk.environment.SimpleEnvironmentProvider).
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
    """

    ENCODING = "utf-8"

    def __init__(
        self,
        dbpath,
        subset=None,
        available_properties=None,
        load_only=None,
        units=None,
        atomref=None,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        centering_function=get_center_of_mass,
        protected=False,
    ):
        # checks and warnings
        # check if database has a valid suffix
        if not dbpath.endswith(".db"):
            raise AtomsDataError(
                "Invalid dbpath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )
        # deprecation error for use of subset argument
        if subset is not None:
            raise AtomsDataError(
                "The subset argument is deprecated and can not be used anymore! "
                "Please use spk.data.partitioning.create_subset or "
                "spk.data.AtomsDataSubset to build subsets."
            )
        # deprecation warning for available_properties argument
        if available_properties is not None:
            warnings.warn(
                "The use of available properties is deprecated. Available properties "
                "are handled automatically since spk 0.3.2!",
                DeprecationWarning,
            )
        # deprecation warning for units
        if type(units) == list:
            warnings.warn(
                "The usage of units has changed. Please provide a dict with "
                "property: unit as input. The use of units as a list is deprecated and "
                "only works when provided with available_properties. Future versions "
                "will not support unit lists anymore!",
                DeprecationWarning,
            )
            if available_properties is None:
                raise AtomsDataError(
                    "No available properties defined. Please set the available "
                    "properties argument!"
                )
            units = dict(zip(available_properties, units))

        # set arguments
        self.dbpath = dbpath
        self.protected = protected
        self._load_only = load_only
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centering_function = centering_function

        # create new database, if not os.path.exists
        if not os.path.exists(self.dbpath):
            with connect(self.dbpath) as conn:
                conn.metadata = dict(
                    available_properties=None,
                    spk_version=spk.__version__,
                    ase_version=ase.__version__,
                    atomref=atomref,
                    units=units,
                )

        # build metadata
        self._metadata = None
        self._initialize_metadata()

        # check if database requires update
        self._update_database()

    def _initialize_metadata(self, units=None, atomref=None):
        with connect(self.dbpath) as conn:
            metadata = conn.metadata

        # add units to metadata
        if "units" not in metadata.keys():
            metadata["units"] = None
        if units is not None:
            # set metadata units, if units not set yet
            if metadata["units"] is None:
                metadata["units"] = units
            # check if stored units match the provided units
            else:
                if units is not None and metadata["units"] != units:
                    raise AtomsDataError(
                        "The units argument does not match the units in the metadata "
                        "of the database!"
                    )

        # add atomref
        if "atomref" not in metadata.keys():
            metadata["atomref"] = None
        if units is not None:
            # set metadata units, if units not set yet
            if metadata["atomref"] is None:
                metadata["atomref"] = atomref
            # check if stored units match the provided units
            else:
                if atomref is not None and metadata["atomref"] != atomref:
                    raise AtomsDataError(
                        "The atomref argument does not match the units in the metadata "
                        "of the database!"
                    )

        # add available_properties
        if "available_properties" not in metadata.keys():
            metadata["available_properties"] = None
        if metadata["available_properties"] is None:
            # build available properties as intersection of the properties of all
            # datapoints
            with connect(self.dbpath) as conn:
                for i in range(conn.__len__()):
                    data = conn.get(i + 1).data
                    if i == 0:
                        metadata["available_properties"] = set(data.keys())
                    else:
                        metadata["available_properties"] = metadata[
                            "available_properties"
                        ].intersection(data.keys())
        elif type(metadata["available_properties"]) == list:
            metadata["available_properties"] = set(metadata["available_properties"])

        # add spk-version
        if "spk_version" not in metadata.keys():
            metadata["spk_version"] = "unknown"
        # add ase version
        if "ase_version" not in metadata.keys():
            metadata["ase_version"] = "unknown"

        # set metadata in class and in database (if not protected)
        self.set_metadata(metadata)

    @property
    def available_properties(self):
        return self.get_metadata("available_properties")

    @property
    def load_only(self):
        if self._load_only is None:
            return self.available_properties
        return self._load_only

    # metadata
    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        if key is None:
            return self._metadata
        elif key not in self._metadata.keys():
            raise AtomsDataError("{} is not a property of the metadata!".format(key))
        return self._metadata[key]

    def set_metadata(self, metadata=None, **kwargs):
        """
        Sets the metadata dictionary of the ASE db.

        Args:
            metadata (dict): dictionary of metadata for the ASE db
            kwargs: further key-value pairs for convenience
        """

        # merge all metadata
        if metadata is not None:
            metadata.update(kwargs)

        # update class metadata
        self._metadata = metadata

        # set new metadata in database, if not protected
        if not self.protected:
            with connect(self.dbpath) as conn:
                # transform sets to list, in order to pickle
                writable_metadata = {
                    k: (list(v) if type(v) == set else v) for k, v in metadata.items()
                }
                conn.metadata = writable_metadata

    def update_metadata(self, **data):
        metadata = self.get_metadata()
        metadata.update(data)
        self.set_metadata(metadata)

    # get atoms and properties
    def get_properties(self, idx, load_only=None):
        """
        Return property dictionary at given index.

        Args:
            idx (int): data index
            load_only (sequence or None): subset of available properties to load

        Returns:
            at (ase.Atoms): atoms object
            properties (dict): dictionary with molecular properties

        """
        # use all available properties if nothing is specified
        if load_only is None:
            load_only = self.available_properties

        # read from ase-database
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for pname in load_only:
            properties[pname] = row.data[pname]

        # extract/calculate structure
        properties = _convert_atoms(
            at,
            environment_provider=self.environment_provider,
            collect_triples=self.collect_triples,
            centering_function=self.centering_function,
            output=properties,
        )

        return at, properties

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        with connect(self.dbpath) as conn:
            row = conn.get(idx + 1)
        at = row.toatoms()
        return at

    # add systems
    def add_system(self, atoms, **properties):
        """
        Add atoms data to the dataset.

        Args:
            atoms (ase.Atoms): system composition and geometry
            **properties: properties as key-value pairs. Keys have to match the
                `available_properties` of the dataset.

        """
        if self.protected:
            raise AtomsDataError("The database is protected. No systems can be added!")

        with connect(self.dbpath) as conn:
            self._add_system(conn, atoms, **properties)

        # update metadata
        available_properties = self.get_metadata("available_properties")
        if available_properties is None:
            available_properties = set(properties.keys())
        else:
            available_properties = available_properties.intersection(properties.keys())
        self.update_metadata(available_properties=available_properties)

    def add_systems(self, atoms_list, property_list):
        """
        Add atoms data to the dataset.

        Args:
            atoms_list (list of ase.Atoms): system composition and geometry
            property_list (list): Properties as list of key-value pairs in the same
                order as corresponding list of `atoms`.
                Keys have to match the `available_properties` of the dataset.

        """
        if self.protected:
            raise AtomsDataError("The database is protected. No systems can be added!")

        with connect(self.dbpath) as conn:
            for at, prop in zip(atoms_list, property_list):
                self._add_system(conn, at, **prop)

        # update metadata
        available_properties = self.get_metadata("available_properties")
        if available_properties is None:
            available_properties = set(property_list[0].keys())
        for data in property_list[1:]:
            available_properties = available_properties.intersection(data.keys())
        self.update_metadata(available_properties=available_properties)

    # deprecated
    def create_subset(self, subset):
        warnings.warn(
            "create_subset is deprecated! Please use "
            "spk.data.partitioning.create_subset.",
            DeprecationWarning,
        )
        from .partitioning import create_subset

        return create_subset(self, subset)

    # __functions__
    def __len__(self):
        with connect(self.dbpath) as conn:
            return conn.count()

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatAtomsData([self, other])

    # private methods
    def _add_system(self, conn, atoms, **properties):

        conn.write(atoms, data=properties)

    def _update_database(self):
        update_helper = spk.utils.DataBaseUpdater(self.dbpath, self._metadata)
        update_helper.update()

        # update metadata
        with connect(self.dbpath) as conn:
            self._metadata = conn.metadata


class ConcatAtomsData(ConcatDataset):
    r"""
    Dataset as a concatenation of multiple atomistic datasets.
    Args:
        datasets (sequence): list of datasets to be concatenated
    """

    def __init__(self, datasets, load_only=None):
        # checks
        for dataset in datasets:
            if not any(
                [
                    isinstance(dataset, dataset_class)
                    for dataset_class in [AtomsData, AtomsDataSubset, ConcatDataset]
                ]
            ):
                raise AtomsDataError(
                    "{} is not an instance of AtomsData, AtomsDataSubset or "
                    "ConcatAtomsData!".format(dataset)
                )

        # initialize
        super(ConcatAtomsData, self).__init__(datasets)

        # build metadata
        self._metadata = self._initialize_metadata()

        # protect datasets
        self.protected = True
        for dataset in self.datasets:
            dataset.protected = True

        # define load only
        if load_only is not None:
            if type(load_only) is list:
                load_only = set(load_only)

            if not load_only.issubset(self.get_metadata("available_properties")):
                raise AtomsDataError(
                    "The selected load_only properties are not in the available "
                    "properties!"
                )
        else:
            load_onlys = [
                dataset.load_only if dataset.load_only is not None else
                dataset.available_properties for dataset in self.datasets
            ]
            load_only = set.intersection(*load_onlys)
        self._load_only = load_only

    @property
    def load_only(self):
        return self._load_only

    @property
    def available_properties(self):
        return self.get_metadata("available_properties")

    def _initialize_metadata(self):
        metadatas = [dataset.get_metadata() for dataset in self.datasets]

        # get available_properties
        available_properties = set.intersection(
            *[meta["available_properties"] for meta in metadatas]
        )

        # get units
        units = [meta["units"] for meta in metadatas if meta["units"] is not None]
        if len(units) == 0:
            units = None
        elif len(units) == 1:
            units = units[0]
        else:
            if all([units[0]==u for u in units]):
                units = units[0]
            else:
                raise AtomsDataError(
                    "Datasets with different units can not be concatenated. Please"
                    " check the units!"
                )

        # get atomref
        atomrefs = [meta["atomref"] for meta in metadatas]
        if all([atomrefs[0] == a for a in atomrefs]):
            atomref = atomrefs[0]
        else:
            raise AtomsDataError(
                "Datasets with different atomrefs can not be concatenated! Please fix "
                "the atomrefs in the metadata before concatenating the datasets."
            )

        # collect metadata
        metadata = dict(
            atomref=atomref,
            units=units,
            available_properties=available_properties,
            spk_version=spk.__version__,
            ase_version=ase.__version__,
        )

        return metadata

    def get_metadata(self, key=None):
        """
        Returns an entry from the metadata dictionary of the ASE db.

        Args:
            key: Name of metadata entry. Return full dict if `None`.

        Returns:
            value: Value of metadata entry or full metadata dict, if key is `None`.

        """
        if key is None:
            return self._metadata
        elif key not in self._metadata.keys():
            raise AtomsDataError("{} is not a property of the metadata!".format(key))
        return self._metadata[key]

    def get_properties(self, idx, load_only=None):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_properties(sample_idx, load_only)

    def set_load_only(self, load_only):
        # check if properties are available
        for pname in load_only:
            if pname not in self.available_properties:
                raise AtomsDataError(
                    "The property '{}' is not an available property and can therefore "
                    "not be loaded!".format(pname)
                )

        # update load_only parameter
        self._load_only = set(load_only)

    def __getitem__(self, idx):
        _, properties = self.get_properties(idx, self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatAtomsData([self, other])


class AtomsDataSubset(Subset):
    r"""
    Subset of an atomistic dataset at specified indices.
    Arguments:
        dataset (torch.utils.data.Dataset): atomistic dataset
        indices (sequence): subset indices
    """

    def __init__(self, dataset, indices):
        super(AtomsDataSubset, self).__init__(dataset, indices)
        self._load_only = None

    @property
    def available_properties(self):
        return self.dataset.available_properties

    @property
    def load_only(self):
        if self._load_only is None:
            return self.dataset.load_only
        return self._load_only

    def get_metadata(self, key=None):
        return self.dataset.get_metadata(key=key)

    def get_properties(self, idx, load_only=None):
        return self.dataset.get_properties(idx, load_only)

    def set_load_only(self, load_only):
        # check if properties are available
        for pname in load_only:
            if pname not in self.available_properties:
                raise AtomsDataError(
                    "The property '{}' is not an available property and can therefore "
                    "not be loaded!".format(pname)
                )

        # update load_only parameter
        self._load_only = list(load_only)

    # deprecated
    def create_subset(self, subset):
        warnings.warn(
            "create_subset is deprecated! Please use "
            "spk.data.partitioning.create_subset.",
            DeprecationWarning,
        )
        from .partitioning import create_subset

        return create_subset(self, subset)

    def __getitem__(self, idx):
        _, properties = self.get_properties(self.indices[idx], self.load_only)
        properties["_idx"] = np.array([idx], dtype=np.int)

        return torchify_dict(properties)

    def __add__(self, other):
        return ConcatAtomsData([self, other])


def _convert_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=None,
    output=None,
):
    """
    Helper function to convert ASE atoms object to SchNetPack input format.

    Args:
        atoms (ase.Atoms): Atoms object of molecule
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        centering_function (callable or None): Function for calculating center of
            molecule (center of mass/geometry/...). Center will be subtracted from
            positions.
        output (dict): Destination for converted atoms, if not None

    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.

    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    inputs[Properties.Z] = atoms.numbers.astype(np.int)
    positions = atoms.positions.astype(np.float32)
    if centering_function:
        positions -= centering_function(atoms)
    inputs[Properties.R] = positions

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    inputs[Properties.neighbors] = nbh_idx.astype(np.int)

    # Get cells
    inputs[Properties.cell] = np.array(atoms.cell.array, dtype=np.float32)
    inputs[Properties.cell_offset] = offsets.astype(np.float32)

    # If requested get neighbor lists for triples
    if collect_triples:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        inputs[Properties.neighbor_pairs_j] = nbh_idx_j.astype(np.int)
        inputs[Properties.neighbor_pairs_k] = nbh_idx_k.astype(np.int)

        inputs[Properties.neighbor_offsets_j] = offset_idx_j.astype(np.int)
        inputs[Properties.neighbor_offsets_k] = offset_idx_k.astype(np.int)

    return inputs


def torchify_dict(property_dict):
    torch_properties = {}
    for pname, prop in property_dict.items():
        if prop.dtype in [np.int, np.int32, np.int64]:
            torch_properties[pname] = torch.LongTensor(prop)
        elif prop.dtype in [np.float, np.float32, np.float64]:
            torch_properties[pname] = torch.FloatTensor(prop)
        else:
            raise AtomsDataError(
                "Invalid datatype {} for property {}!".format(type(prop), pname)
            )
    return torch_properties


class AtomsConverter:
    """
    Convert ASE atoms object to an input suitable for the SchNetPack
    ML models.

    Args:
        environment_provider (callable): Neighbor list provider.
        collect_triples (bool, optional): Set to True if angular features are needed.
        device (str): Device for computation (default='cpu')
        add_batch_dimension (bool): return converted data as batch shape if True
    """

    def __init__(
        self,
        environment_provider=SimpleEnvironmentProvider(),
        collect_triples=False,
        device=torch.device("cpu"),
        add_batch_dimension=True,
    ):
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.add_batch_dimension = add_batch_dimension
        # Get device
        self.device = device

    def __call__(self, atoms):
        """
        Args:
            atoms (ase.Atoms): Atoms object of molecule

        Returns:
            dict of torch.Tensor: Properties including neighbor lists and masks
                reformated into SchNetPack input format.
        """
        inputs = _convert_atoms(atoms, self.environment_provider, self.collect_triples)
        inputs = torchify_dict(inputs)

        # Calculate masks
        inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
        mask = inputs[Properties.neighbors] >= 0
        inputs[Properties.neighbor_mask] = mask.float()
        inputs[Properties.neighbors] = (
            inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
        )

        if self.collect_triples:
            mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
            mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
            mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
            inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

        # Add batch dimension and move to CPU/GPU
        for key, value in inputs.items():
            if self.add_batch_dimension:
                value = value.unsqueeze(0)
            inputs[key] = value.to(self.device)

        return inputs
