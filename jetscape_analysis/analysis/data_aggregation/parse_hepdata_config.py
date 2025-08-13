"""Parse the HEPdata configuration based on the available parameters.

The idea here is to provide a consistent parsing standard while making it as
easy as possible to deal with the immense combinators of full data aggregation.

The specification is as follows:
- parameters are shared settings
- combinations are parameters that vary within that group
- Lists in the parameters field mean “all combinations of these values”

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import csv
import itertools
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from jetscape_analysis.base import helpers

logger = logging.getLogger(__name__)


def cartesian_expand(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand parameters via a cartesian product.

    Principles used for expanding parameters:

    - A list of scalars means multiple values to expand.
    - A list of lists means each sub-list is a single atomic value (converted to tuple). For example,
      a pairing of centrality.
    - All atomic values are hashable (lists converted to tuples).

    Args:
        params: Parameters to combine
    Returns:
        A list of dictionaries containing parameter -> [value] pairs
    """
    keys = list(params.keys())
    values: list[list[Any]] = []
    for v in params.values():
        if isinstance(v, list):
            # Case: list of lists -> each sublist should be passed along (i.e. atomic)
            # NOTE: We convert to tuple so that the values are hashable.
            if v and isinstance(v[0], list):
                values.append([tuple(x) if isinstance(x, list) else x for x in v])
            else:
                # Multi-values in list, so while treat each one as a value (i.e. a list of scalars)
                values.append(v)
        else:
            # If only a single value, it needs to be a list to work well with itertools.product
            values.append([v])

    # Cartesian product expansion
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo, strict=True)))
    return combos


def expand_group(group: dict[str, Any], inherited_params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Recursively expand a group into a flat list of mappings.

    Each mapping has:
        - parameters: dict[str, Any] (fully resolved, scalar values)
        - table: int
        - entry: int


    """
    inherited_params = inherited_params or {}

    # Merge inherited parameters with this group's parameters
    group_params: dict[str, Any] = group.get("parameters", {})
    merged_params: dict[str, Any] = {**inherited_params, **group_params}

    # Expand any list-valued parameters into multiple dicts
    expanded_param_sets: list[dict[str, Any]] = cartesian_expand(merged_params)

    results: list[dict[str, Any]] = []

    if "combinations" in group:
        # Recurse into each combination
        for param_set in expanded_param_sets:
            for child in group["combinations"]:
                results.extend(expand_group(child, param_set))
    else:
        # This is a final mapping (must have table + entry)
        if "table" not in group or "entry" not in group:
            msg = f"Final mapping missing table/entry: {group}"
            raise ValueError(msg)
        for param_set in expanded_param_sets:
            results.append({"parameters": param_set, "table": int(group["table"]), "entry": int(group["entry"])})

    return results


def expand_yaml(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Expand the grouped_spectra for each dataset (AA, pp, etc.)
    into a flat list of mappings.
    """
    expanded: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, dataset in data.get("data", {}).items():
        hepdata: dict[str, Any] = dataset.get("hepdata", {})
        groups: list[dict[str, Any]] = hepdata.get("grouped_spectra", [])
        all_mappings: list[dict[str, Any]] = []
        for group in groups:
            all_mappings.extend(expand_group(group))
        expanded[dataset_name] = all_mappings
    return expanded


def validate_no_duplicates(expanded: dict[str, list[dict[str, Any]]]) -> None:
    """
    Validate that there are no duplicate parameter sets within each dataset.
    """
    for dataset, mappings in expanded.items():
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for m in mappings:
            key = tuple(sorted(m["parameters"].items()))
            if key in seen:
                msg = f"Duplicate parameters in dataset '{dataset}': {m['parameters']}"
                raise ValueError(msg)
            seen.add(key)


def validate_missing_combinations(
    expanded: dict[str, list[dict[str, Any]]], expected_grid: dict[str, list[Any]]
) -> None:
    """
    Validate that all combinations from the expected parameter grid
    are present in the expanded mapping, and print a grouped summary.
    """
    # Normalize expected_grid so all atomic values are hashable
    normalized_grid: dict[str, list[Any]] = {}
    for key, values in expected_grid.items():
        normalized_values = [tuple(v) if isinstance(v, list) else v for v in values]
        normalized_grid[key] = normalized_values

    expected_combos: list[tuple[Any, ...]] = list(itertools.product(*normalized_grid.values()))
    expected_keys: list[str] = list(expected_grid.keys())

    for dataset, mappings in expanded.items():
        found_set: set[tuple[Any, ...]] = {tuple(m["parameters"].get(k) for k in expected_keys) for m in mappings}
        missing: list[dict[str, Any]] = []
        for combo in expected_combos:
            if combo not in found_set:
                missing.append(dict(zip(expected_keys, combo, strict=True)))

        if missing:
            logger.warning(f"Missing {len(missing)} combinations for dataset '{dataset}':")
            for m in missing:
                logger.warning(f"  - {m}")

            # Grouped summary by parameter values
            logger.info(f"[SUMMARY] Missing combinations breakdown for '{dataset}':")
            for key in expected_keys:
                counts: Counter[Any] = Counter(m[key] for m in missing)
                for val, count in counts.items():
                    logger.info(f"  {key} = {val}: {count} missing")
            logger.info("")


def write_csv(expanded: dict[str, list[dict[str, Any]]], filename: Path) -> None:
    """
    Write expanded mappings to a CSV file for debugging.
    """
    # Determine all parameter keys
    all_params: set[str] = set()
    for mappings in expanded.values():
        for m in mappings:
            all_params.update(m["parameters"].keys())
    param_keys: list[str] = sorted(all_params)

    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["dataset", *param_keys, "table", "entry"]
        writer.writerow(header)
        for dataset, mappings in expanded.items():
            for m in mappings:
                row = [dataset] + [m["parameters"].get(k) for k in param_keys] + [m["table"], m["entry"]]
                writer.writerow(row)


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)
    # Example usage
    yaml_file = Path("example.yaml")
    with yaml_file.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    expanded = expand_yaml(config)
    validate_no_duplicates(expanded)

    # Example expected parameter grid (provided separately in your workflow)
    expected_grid: dict[str, list[Any]] = {
        "jet_R": [0.2, 0.3],
        "soft_drop": ["z_cut_02_beta_0"],
        "centrality": [[0, 10], [10, 20]],
        "jet_pt": [[100, 200], [200, 300]],
    }
    validate_missing_combinations(expanded, expected_grid)

    # Output CSV for debugging
    write_csv(expanded, Path("expanded_mapping.csv"))

    logger.info("✅ Expansion complete. CSV written to expanded_mapping.csv")
