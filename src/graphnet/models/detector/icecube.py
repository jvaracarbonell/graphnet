"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os
import numpy as np
from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeKaggle(Detector):
    """`Detector` class for Kaggle Competition."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["x", "y", "z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "x": self._xyz,
            "y": self._xyz,
            "z": self._xyz,
            "time": self._time,
            "charge": self._charge,
            "auxiliary": self._identity,
        }
        return feature_map

    def _xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 3.0


class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xy,
            "dom_y": self._dom_xy,
            "dom_z": self._dom_z,
            "dom_time": self._dom_time,
            "charge": self._identity,
            "rde": self._rde,
            "pmt_area": self._pmt_area,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350.0) / 100.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return ((x / 1.05e04) - 1.0) * 20.0

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeUpgrade(Detector):
    """`Detector` class for IceCube-Upgrade."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._identity,
            "pmt_area": self._pmt_area,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "hlc": self._identity,
        }

        return feature_map

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x / 2e04) - 1.0

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x) / 2.0

    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x - 50.0) / 50.0

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 20.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return (x - 60.0) / 60.0

    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 130.0

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05


class IceCubeGen2SUM(Detector):
    """`Detector` class for IceCube-Upgrade."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "IceCubeGen2SUM.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "sum_4": self._sum_4,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "sum_1": self._sum_charges,
            "sum_2": self._sum_charges,
            "sum_3": self._sum_charges,
            "sum_5": self._sum_6, # features 6 and 5 are very related
            "sum_6": self._sum_6,
            "sum_7": self._sum_7,
            "sum_8": self._sum_8,
            "sum_9": self._sum_9,
            "is_saturated_dom": self._identity,
            "is_errata_dom": self._identity,
            
        }

        return feature_map
    
    def _sum_charges(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 1.05838) / (2.51312 - 0.70168) # In case I want to scale all charges together to preserve their relations

    def _sum_1(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 1.05838) / (2.51312 - 0.70168)
    
    def _sum_2(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 0.99207) / (2.08278 - 0.68907)

    def _sum_3(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 0.84017) / (1.29265 - 0.63489)
    
    def _sum_4(self, x: torch.tensor) -> torch.tensor:
        return (x - 9517.3786 ) / ( 12895.6868 - 7085.2987)
    
    def _sum_5(self, x: torch.tensor) -> torch.tensor:
        return (x - 9517.3786 ) / ( 12895.6868 - 7085.2987)
    
    def _sum_6(self, x: torch.tensor) -> torch.tensor:
        return (x - 9720.6075 ) / ( 13134.1827 - 7280.60751)
    
    def _sum_7(self, x: torch.tensor) -> torch.tensor:
        return (x - 10517.3211 ) / ( 14437.6869 - 7848.9976)
    
    def _sum_8(self, x: torch.tensor) -> torch.tensor:
        return (x - 9799.2268 ) / ( 13242.8468 - 7346.1734)

    def _sum_9(self, x: torch.tensor) -> torch.tensor:
        return (x-26.0379) / 355.3265 

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return (x + 2000) / 4000
    
    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x / 1120)

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 15.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return x / 80.0 
    
    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 171.0
    

class IceCubeGen2SUMCUT(Detector):
    """`Detector` class for IceCube-Gen2
    At least 200 PMTs with hits after noise cleaning."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "IceCubeGen2SUM.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "sum_4": self._sum_4,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "sum_1": self._sum_charges,
            "sum_2": self._sum_charges,
            "sum_3": self._sum_charges,
            "sum_5": self._sum_6, # features 6 and 5 are very related
            "sum_6": self._sum_6,
            "sum_7": self._sum_7,
            "sum_8": self._sum_8,
            "sum_9": self._sum_9,
            "is_saturated_dom": self._identity,
            "is_errata_dom": self._identity,
            
        }

        return feature_map
    
    def _sum_charges(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 1.17508) / (2.6590939 - 0.72017) # In case I want to scale all charges together to preserve their relations

    def _sum_1(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 1.17508) / (2.6590939 - 0.72017)
    
    def _sum_2(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 1.0655370) / (2.1975092 - 0.70513)

    def _sum_3(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) - 0.8547416) / (1.3432940 - 0.642336)
    
    def _sum_4(self, x: torch.tensor) -> torch.tensor:
        return (x - 700.30891 ) / ( 2265.172313 + 583.82758)
    
    def _sum_5(self, x: torch.tensor) -> torch.tensor:
        return (x - 795.696289 ) / ( 2332.62740 + 515.5564)
    
    def _sum_6(self, x: torch.tensor) -> torch.tensor:
        return (x - 963.982735 ) / ( 2501.91262 + 394.569688)
    
    def _sum_7(self, x: torch.tensor) -> torch.tensor:
        return (x - 1745.155413 ) / ( 3696.83540 - 52.16607)
    
    def _sum_8(self, x: torch.tensor) -> torch.tensor:
        return (x - 1052.8429361 ) / ( 2597.53240 + 326.1857300)

    def _sum_9(self, x: torch.tensor) -> torch.tensor:
        return (x-61.36591196) / 379.352222 

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return (x + 2000) / 4000
    
    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x / 1120)

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 15.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return x / 80.0 
    
    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 171.0
    

class IceCubeGen2SUMCUTRlogl(Detector):
    """`Detector` class for IceCube-Gen2
    At least 200 PMTs with hits after noise cleaning."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "IceCubeGen2SUM.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "sum_4": self._sum_4,
            "string": self._string,
            "pmt_number": self._pmt_number,
            "dom_number": self._dom_number,
            "pmt_dir_x": self._identity,
            "pmt_dir_y": self._identity,
            "pmt_dir_z": self._identity,
            "dom_type": self._dom_type,
            "sum_1": self._sum_charges,
            "sum_2": self._sum_charges,
            "sum_3": self._sum_charges,
            "sum_5": self._sum_6, # features 6 and 5 are very related
            "sum_6": self._sum_6,
            "sum_7": self._sum_7,
            "sum_8": self._sum_8,
            "sum_9": self._sum_9,
            "is_saturated_dom": self._identity,
            "is_errata_dom": self._identity,
            
        }

        return feature_map
    
    def _sum_charges(self, x: torch.tensor) -> torch.tensor:
        return (torch.log10(x+1) -  1.463408783) / (2.94114116 - 0.76422004) # In case I want to scale all charges together to preserve their relations
    
    def _sum_4(self, x: torch.tensor) -> torch.tensor:
        return (x - 797.30239863 ) / ( 2274.400887 + 398.20205)
    
    def _sum_5(self, x: torch.tensor) -> torch.tensor:
        return (x - 900.432466 ) / ( 2347.5502374 + 315.34718)
    
    def _sum_6(self, x: torch.tensor) -> torch.tensor:
        return (x - 1079.899473 ) / ( 2525.06321067 + 178.256262)
    
    def _sum_7(self, x: torch.tensor) -> torch.tensor:
        return (x - 1957.36797566 ) / ( 3812.841704488 - 378.5153496)
    
    def _sum_8(self, x: torch.tensor) -> torch.tensor:
        return (x - 1176.305922086 ) / ( 2629.0924346 + 93.8063492)

    def _sum_9(self, x: torch.tensor) -> torch.tensor:
        return (x-152.5072668) / 422.900994

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return (x + 2000) / 4000
    
    def _string(self, x: torch.tensor) -> torch.tensor:
        return (x / 1120)

    def _pmt_number(self, x: torch.tensor) -> torch.tensor:
        return x / 15.0

    def _dom_number(self, x: torch.tensor) -> torch.tensor:
        return x / 80.0 
    
    def _dom_type(self, x: torch.tensor) -> torch.tensor:
        return x / 171.0