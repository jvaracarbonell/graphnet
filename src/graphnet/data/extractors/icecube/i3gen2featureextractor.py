"""I3Extractor class(es) for extracting specific, reconstructed features."""

from typing import TYPE_CHECKING, Any, Dict, List
from graphnet.data.extractors.icecube.i3featureextractor import I3FeatureExtractor
from graphnet.data.extractors.icecube.i3extractor import I3Extractor
from graphnet.data.extractors.icecube.utilities.frames import (
    get_om_keys_and_pulseseries, get_dnn_values,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import (
        icetray,
        dataclasses,
    )  

class I3FeatureExtractorIceCubeGen2SUM(I3FeatureExtractor):
    """Class for extracting features anf pulse summary statitics for IceCube-Gen2.
    The summary statistics were calculated using a modified ic3-data version from Mirco's original"""

    def __call__(self, frame: "icetray.I3Frame") -> Dict[str, List[Any]]:
        """Extract reconstructed features from `frame`.

        Args:
            frame: Physics (P) I3-frame from which to extract reconstructed
                features.

        Returns:
            Dictionary of reconstructed features for all pulses in `pulsemap`,
                in pure-python format.
        """
        padding_value: float = -1.0
        output: Dict[str, List[Any]] = {
            "pmt_dir_x": [],
            "pmt_dir_y": [],
            "pmt_dir_z": [],
            "dom_x": [],
            "dom_y": [],
            "dom_z": [],
            "sum_1": [],
            "sum_2": [],
            "sum_3": [],
            "sum_4": [],
            "sum_5": [],
            "sum_6": [],
            "sum_7": [],
            "sum_8": [],
            "sum_9": [],
            "is_bright_dom": [],
            "is_bad_dom": [],
            "is_saturated_dom": [],
            "is_errata_dom": [],
            "event_time": [],
            "string": [],
            "pmt_number": [],
            "dom_number": [],
            "dom_type": [],
        }
        # Get OM data
        if self._pulsemap in frame:
            om_keys, data = get_dnn_values(
                frame,
                "dnn_global_values",#self._pulsemap,
                self._calibration,
            )
        else:
            self.warning_once(f"Pulsemap {self._pulsemap} not found in frame.")
            return output

        # Added these :
        bright_doms = None
        bad_doms = None
        saturation_windows = None
        calibration_errata = None
        gen2_saturation = None
        if "BrightDOMs" in frame:
            bright_doms = frame.Get("BrightDOMs")

        if "BadDomsList" in frame:
            bad_doms = frame.Get("BadDomsList")

        if "SaturationWindows" in frame:
            saturation_windows = frame.Get("SaturationWindows")
            
        if "Gen2SaturationMap" in frame:
            gen2_saturation = frame.Get("Gen2SaturationMap")

        if "CalibrationErrata" in frame:
            calibration_errata = frame.Get("CalibrationErrata")

        event_time = frame["I3EventHeader"].start_time.mod_julian_day_double

        for om_key in om_keys:
           
            # Common values for each OM
            x = self._gcd_dict[om_key].position.x
            y = self._gcd_dict[om_key].position.y
            z = self._gcd_dict[om_key].position.z
            #area = self._gcd_dict[om_key].area
            #rde = self._get_relative_dom_efficiency(
            #    frame, om_key, padding_value
            #)
            
            string = om_key[0]
            dom_number = om_key[1]
            pmt_number = om_key[2]
            dom_type = self._gcd_dict[om_key].omtype

            # DOM flags
            if bright_doms:
                is_bright_dom = 1 if om_key in bright_doms else 0
            else:
                is_bright_dom = int(padding_value)

            if bad_doms:
                is_bad_dom = 1 if om_key in bad_doms else 0
            else:
                is_bad_dom = int(padding_value)

            if saturation_windows:
                is_saturated_dom = 1 if om_key in saturation_windows else 0
            else:
                is_saturated_dom = int(padding_value)

            if gen2_saturation and is_saturated_dom != 1:
                is_saturated_dom = 1 if om_key in gen2_saturation else 0
            else:
                is_saturated_dom = int(padding_value)
            
            if calibration_errata:
                is_errata_dom = 1 if om_key in calibration_errata else 0
            else:
                is_errata_dom = int(padding_value)

            # Loop over pulses for each OM
            #pulses = data[om_key]
            
            # Add pulse summary statistics
            pulse = data[om_key]

            output["sum_1"].append(
                pulse[0])
            output["sum_2"].append(
                pulse[1])
            output["sum_3"].append(
                pulse[2])
            output["sum_4"].append(
                pulse[3])
            output["sum_5"].append(
                pulse[4])
            output["sum_6"].append(
                pulse[5])
            output["sum_7"].append(
                pulse[6])
            output["sum_8"].append(
                pulse[7])
            output["sum_9"].append(
                pulse[8])
    
            output["dom_x"].append(x)
            output["dom_y"].append(y)
            output["dom_z"].append(z)
            
            # ID's
            output["string"].append(string)
            output["pmt_number"].append(pmt_number)
            output["dom_number"].append(dom_number)
            output["dom_type"].append(dom_type)
           
            # DOM flags
            output["is_bright_dom"].append(is_bright_dom)
            output["is_bad_dom"].append(is_bad_dom)
            output["is_saturated_dom"].append(is_saturated_dom)
            output["is_errata_dom"].append(is_errata_dom)
            output["event_time"].append(event_time)
            
            

        for om_key in om_keys:
            # Common values for each OM
            pmt_dir_x = self._gcd_dict[om_key].orientation.x
            pmt_dir_y = self._gcd_dict[om_key].orientation.y
            pmt_dir_z = self._gcd_dict[om_key].orientation.z

            # Loop over pulses for each OM

            output["pmt_dir_x"].append(pmt_dir_x)
            output["pmt_dir_y"].append(pmt_dir_y)
            output["pmt_dir_z"].append(pmt_dir_z)
        
        return output


