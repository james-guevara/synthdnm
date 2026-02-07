"""Configuration loader for SynthDNM pipeline."""

from pathlib import Path
from typing import Optional, Union

# Try Python 3.11+ tomllib, fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


DEFAULT_CONFIG = {
    "columns": {
        "id": [
            "#CHROM", "POS", "REF", "ALT", "ALT_specific",
            "SAMPLE", "AC", "allele_num",
        ],
        "label": "truth",
        "drop": [],
    },
    "features": {
        "universal": {
            "description": "Trio genotype quality features from FORMAT fields",
            "columns": [
                "child_AR", "min_AR", "max_AR",
                "child_GQ", "min_GQ", "max_GQ",
                "child_DP", "min_DP", "max_DP",
                "child_PL0", "min_PL0", "max_PL0",
                "child_PL1", "min_PL1", "max_PL1",
                "child_PL2", "min_PL2", "max_PL2",
                "child_AB", "father_AB", "mother_AB",
                "indel_flag", "haploid_flag",
            ],
        },
        "gatk": {
            "description": "GATK variant quality metrics",
            "extends": "universal",
            "columns": [
                "INFO_QD", "INFO_FS", "INFO_MQ", "INFO_SOR",
                "INFO_BaseQRankSum", "INFO_MQRankSum", "INFO_ReadPosRankSum",
            ],
        },
        "ssc": {
            "description": "SSC-specific (adds VQSLOD)",
            "extends": "gatk",
            "columns": ["INFO_VQSLOD"],
        },
    },
    "training": {
        "test_size": 0.2,
        "seed": 42,
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "early_stopping_rounds": 50,
    },
}


def load_config(config_path: Optional[Union[Path, str]] = None) -> dict:
    """Load configuration from TOML file, with defaults as fallback.

    Args:
        config_path: Path to features.toml. If None, looks for features.toml
                     in the same directory as this module, then current directory.

    Returns:
        Configuration dictionary with all settings.
    """
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is None:
        module_dir = Path(__file__).parent
        candidates = [module_dir / "features.toml", Path("features.toml")]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        if tomllib is None:
            print("Warning: tomllib/tomli not available, using defaults")
            return config

        with open(config_path, "rb") as f:
            file_config = tomllib.load(f)

        # Deep merge file_config into config
        for section, values in file_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    return config


def get(config: dict, *keys, default=None):
    """Get a nested config value safely.

    Example: get(config, "training", "seed") -> 42
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def resolve_feature_set(config: dict, name: str) -> list[str]:
    """Resolve a feature set by name, following the extends chain.

    Args:
        config: Loaded config dictionary.
        name: Feature set name (e.g., "universal", "gatk", "ssc").

    Returns:
        Flat list of all feature column names for this set.

    Raises:
        KeyError: If the feature set name is not found.
        ValueError: If the extends chain has a cycle.
    """
    features_section = config["features"]

    if name not in features_section:
        available = ", ".join(sorted(features_section.keys()))
        raise KeyError(f"Unknown feature set '{name}'. Available: {available}")

    # Walk the extends chain, collecting columns bottom-up
    columns = []
    visited = set()
    current = name

    while current is not None:
        if current in visited:
            raise ValueError(f"Circular extends chain: {current}")
        visited.add(current)

        entry = features_section[current]
        columns.append(entry.get("columns", []))
        current = entry.get("extends")

    # Reverse so base tier comes first, then extensions
    columns.reverse()
    # Flatten, preserving order and deduplicating
    seen = set()
    result = []
    for col_list in columns:
        for col in col_list:
            if col not in seen:
                seen.add(col)
                result.append(col)

    return result


def list_feature_sets(config: dict) -> dict[str, str]:
    """List available feature sets with their descriptions.

    Returns:
        Dict of {name: description}.
    """
    return {
        name: entry.get("description", "")
        for name, entry in config["features"].items()
    }
