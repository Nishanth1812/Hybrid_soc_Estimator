from pathlib import Path


SIM_CONFIG = {
    "num_healthy": 75,
    "num_degraded": 75,
    "sampling_period_s": 1,
    "duration_s": 10800,
    "temperature_levels_c": [0, 10, 25, 40, 50],
    "seed": 42,
    "spme_options": {"thermal": "lumped"},
    "current_profile_types": ["constant", "pulse", "random", "drive_cycle"],
    "constant_c_rates": [0.5, 1.0, 2.0],
    "max_abs_c_rate": 2.0,
    "initial_soc_range": [0.75, 0.95],
}


DEGRADATION_CONFIG = {
    "levels": {
        0: {
            "name": "healthy",
            "enable_sei": False,
            "lli_fraction": 0.0,
            "resistance_multiplier": 1.0,
        },
        1: {
            "name": "mild",
            "enable_sei": True,
            "lli_fraction": 0.02,
            "resistance_multiplier": 1.12,
        },
        2: {
            "name": "moderate",
            "enable_sei": True,
            "lli_fraction": 0.045,
            "resistance_multiplier": 1.25,
        },
        3: {
            "name": "severe",
            "enable_sei": True,
            "lli_fraction": 0.07,
            "resistance_multiplier": 1.4,
        },
    }
}


PREPROCESSING_CONFIG = {
    "filter_window": 11,
    "filter_polyorder": 2,
    "sequence_length": 100,
    "stride": 1,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "shuffle_seed": 42,
}


BASE_PATH = Path("datasets")
RAW_PATH = BASE_PATH / "raw"
PROCESSED_PATH = BASE_PATH / "processed"
SCALERS_PATH = BASE_PATH / "scalers"
SCALER_PATH = SCALERS_PATH / "input_scaler.pkl"
RAW_METADATA_PATH = RAW_PATH / "metadata.csv"