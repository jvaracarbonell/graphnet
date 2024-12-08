"""Global constants that are used with `graphnet.data`."""


class FEATURES:
    """Namespace for standard names working with `I3FeatureExtractor`."""

    ICECUBE86 = [
        "dom_x",
        "dom_y",
        "dom_z",
        "dom_time",
        "charge",
        "rde",
        "pmt_area",
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE + [
        "string",
        "pmt_number",
        "dom_number",
        "pmt_dir_x",
        "pmt_dir_y",
        "pmt_dir_z",
        "dom_type",
    ]
    ICECUBEGEN2SUM = [
        "dom_x",
        "dom_y",
        "dom_z",
        "sum_4", # first time
        "string",
        "pmt_number",
        "dom_number",
        "pmt_dir_x",
        "pmt_dir_y",
        "pmt_dir_z",
        "dom_type",
        "sum_1",
        "sum_2",
        "sum_3",
        "sum_5",
        "sum_6",
        "sum_7",
        "sum_8",
        "sum_9",
        "is_saturated_dom",
        "is_errata_dom",

    ]
    PROMETHEUS = [
        "sensor_pos_x",
        "sensor_pos_y",
        "sensor_pos_z",
        "t",
    ]
    KAGGLE = ["x", "y", "z", "time", "charge", "auxiliary"]
    LIQUIDO = ["sipm_x", "sipm_y", "sipm_z", "t"]


class TRUTH:
    """Namespace for standard names working with `I3TruthExtractor`."""

    ICECUBE86 = [
        "energy",
        "energy_track",
        "energy_cascade",
        "energy_muon",
        "deposited_energy",
        "position_x",
        "position_y",
        "position_z",
        "azimuth",
        "zenith",
        "pid",
        "elasticity",
        "interaction_type",
        "interaction_time",  # Added for vertex reconstruction
        "inelasticity",
        "stopped_muon",
    ]
    DEEPCORE = ICECUBE86
    UPGRADE = DEEPCORE
    ICECUBEGEN2SUM = ICECUBE86
    PROMETHEUS = [
        "injection_energy",
        "injection_type",
        "injection_interaction_type",
        "injection_zenith",
        "injection_azimuth",
        "injection_bjorkenx",
        "injection_bjorkeny",
        "injection_position_x",
        "injection_position_y",
        "injection_position_z",
        "injection_column_depth",
        "primary_lepton_1_type",
        "primary_hadron_1_type",
        "primary_lepton_1_position_x",
        "primary_lepton_1_position_y",
        "primary_lepton_1_position_z",
        "primary_hadron_1_position_x",
        "primary_hadron_1_position_y",
        "primary_hadron_1_position_z",
        "primary_lepton_1_direction_theta",
        "primary_lepton_1_direction_phi",
        "primary_hadron_1_direction_theta",
        "primary_hadron_1_direction_phi",
        "primary_lepton_1_energy",
        "primary_hadron_1_energy",
        "total_energy",
    ]
    KAGGLE = ["zenith", "azimuth"]
    LIQUIDO = [
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "zenith",
        "azimuth",
        "interaction_time",
        "energy",
        "pid",
    ]
