import schnetpack as spk
from ase.db import connect
import ase
import numpy as np
import os
import warnings
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


__all__ = ["DataBaseUpdater"]


class DataBaseUpdater:
    def __init__(self, db_path, metadata):
        self.db_path = db_path
        self.metadata = metadata

    def update(self):
        # update datatypes to spk-version 0.3.2
        if self.requires_dtype_update():
            logging.info(
                "The database is older than spk 0.3.2 and will be updated to spk "
                "version 0.3.2!"
            )
            warnings.warn(
                "The database is deprecated and will be updated automatically. "
                "The old database is moved to {}.deprecated!".format(self.db_path)
            )

            # read old database
            atoms_list, properties_list = spk.utils.read_deprecated_database(
                self.db_path
            )
            metadata = self.metadata

            # move old database
            os.rename(self.db_path, self.db_path + ".0_3_1.deprecated")

            # write updated database
            metadata["spk_version"] = "0.3.2"
            metadata["ase_version"] = ase.__version__
            with connect(self.db_path) as conn:
                conn.metadata = metadata
                for atoms, properties in tqdm(
                    zip(atoms_list, properties_list),
                    "Updating new database",
                    total=len(atoms_list),
                ):
                    conn.write(atoms, data=properties)

    def requires_dtype_update(self):
        """
        Check if database uses the ase >= 3.19.1 version of storing properties.

        """
        # check if path exists
        if not os.path.exists(self.db_path):
            return False

        with connect(self.db_path) as conn:
            # check if database has entries
            if len(conn) == 0:
                return False

            # if db has at least 1 entry, get the data
            data = conn.get(1).data

        # check byte style deprecation
        if True in [pname.startswith("_dtype_") for pname in data.keys()]:
            return True
        # fallback for properties stored directly in the row
        if True in [type(val) != np.ndarray for val in data.values()]:
            return True

        return False
