"""A CuratedDataset for unit tests."""
from typing import Dict, Any, List, Tuple, Union
import os
from sklearn.model_selection import train_test_split

from graphnet.training.labels import Direction
from graphnet.data import ERDAHostedDataset
from graphnet.data.constants import FEATURES
from graphnet.data.utilities import query_database


class TRIDENTSmall(ERDAHostedDataset):
    """A Dataset for Prometheus simulation of TRIDENT.

    Small version with ~ 1 mill track events.
    """

    # Static Member Variables:
    _pulsemaps = ["photons"]
    _truth_table = "mc_truth"
    _event_truth = [
        "interaction",
        "initial_state_energy",
        "initial_state_type",
        "initial_state_zenith",
        "initial_state_azimuth",
        "initial_state_x",
        "initial_state_y",
        "initial_state_z",
    ]
    _pulse_truth = None
    _features = FEATURES.PROMETHEUS
    _experiment = "TRIDENT Prometheus Simulation"
    _creator = "Rasmus F. Ørsøe"
    _comments = (
        "Contains ~1 million track events."
        " Simulation produced by Stephan Meighen-Berger, "
        "U. Melbourne."
    )
    _available_backends = ["sqlite"]
    _file_hashes = {"sqlite": "F2R8qb8JW7"}
    _citation = None

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments for dataset.

        Args:
            backend: backend of dataset. Either "parquet" or "sqlite"
            features: List of features from user to use as input.
            truth: List of event-level truth form user.

        Returns: Dataset arguments and selections
        """
        dataset_path = os.path.join(self.dataset_dir, "merged.db")

        event_nos = query_database(
            database=dataset_path, query="SELECT event_no FROM mc_truth"
        )

        train_val, test = train_test_split(
            event_nos["event_no"].tolist(),
            test_size=0.10,
            random_state=42,
            shuffle=True,
        )
        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_path,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
            "labels": {
                "direction": Direction(
                    azimuth_key="initial_state_azimuth",
                    zenith_key="initial_state_zenith",
                )
            },
        }

        return dataset_args, train_val, test
