"""Convert the HEPdata format from the first production to the new (2026) format

This is mostly a one-off script - we should only need to do the conversion once.
But it should make it less work for us to switch to the new format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import io
import itertools
import logging
from pathlib import Path
from typing import Any

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from jetscape_analysis.data_curation import data, hepdata_utils, observable

logger = logging.getLogger(__name__)


def generate_entry_name_from_parameters(
    n_pt_bins: int = 1,
    pt_index: int | None = None,
    jet_R: observable.JetRSpec | None = None,
    kappa: observable.AngularitySpec | None = None,
    soft_drop: observable.SoftDropSpec | None = None,
    axis: observable.JetAxisDifferenceSpec | None = None,
    # kwargs just absorbs the rest of the arguments, so they can be passed blindly
    **kwargs: Any,  # noqa: ARG001
) -> str:
    """Generate HEPData v1 entry name from HEPData v2 parameters.

    We basically dump all of the parameters here and just let it sort it out.
    """
    suffix = ""
    # jet_R
    if jet_R is not None:
        suffix += f"_R{jet_R.R}"
    # pt
    pt_suffix = ""
    # We only want to include if we have multiple pt bins
    # NOTE: This might change in the v2 format, but this is correct for the v1 format.
    if pt_index is not None and n_pt_bins > 2:
        pt_suffix = f"_pt{pt_index}"
    # soft drop
    # Generated: hepdata_pp_dir_R0.2_zcut0.2_beta0_WTA_SD
    # In config: hepdata_pp_dir_R0.2_zcut0.2_beta0_WTA_SD_pt0
    if soft_drop is not None:
        suffix += f"_zcut{soft_drop.z_cut:g}_beta{soft_drop.beta:g}"
    # kappa
    if kappa is not None:
        suffix += f"_k{kappa.kappa}"
    # Jet-axis difference
    if axis:
        if axis.grooming_settings:
            suffix += f"_zcut{axis.grooming_settings.z_cut:g}_beta{axis.grooming_settings.beta:g}"
        suffix += f"_{axis.type}"

    return suffix, pt_suffix


def find_hepdata_v1_key_in_block(
    block: dict[str, Any], collision_system: str, centrality_index: int, suffix: str, pt_suffix: str
) -> tuple[str, str] | None:
    """Extracts the HEPData table and name from an HEPData v1 block"""
    # First, check for dir names in config
    dir_key = ""
    if f"hepdata_{collision_system}_dir{suffix}" in block:
        dir_key = f"hepdata_{collision_system}_dir{suffix}"
    elif f"hepdata_{collision_system}_dir{suffix}{pt_suffix}" in block:
        dir_key = f"hepdata_{collision_system}_dir{suffix}{pt_suffix}"
    elif f"hepdata_{collision_system}_dir" in block:
        dir_key = f"hepdata_{collision_system}_dir"
    else:
        logger.info(f"hepdata_{collision_system}_dir{suffix}{pt_suffix} not found!")
        return ("", "")

    # Check for hist names in config
    g_key = ""
    if f"hepdata_{collision_system}_gname{suffix}" in block:
        g_key = f"hepdata_{collision_system}_gname{suffix}"
    elif f"hepdata_{collision_system}_gname{suffix}{pt_suffix}" in block:
        g_key = f"hepdata_{collision_system}_gname{suffix}{pt_suffix}"
    elif f"hepdata_{collision_system}_gname" in block:
        g_key = f"hepdata_{collision_system}_gname"
    else:
        logger.info(f"hepdata_{collision_system}_gname{suffix} not found!")
        return ("", "")

    # Get the appropriate centrality entry in the dir/hist list
    if type(block[dir_key]) is list:
        # If fewer entries than the observable's centrality, skip
        if centrality_index > len(block[dir_key]) - 1:
            return ("", "")

        dir_name = block[dir_key][centrality_index]
        g_name = block[g_key]

    elif type(block[g_key]) is list:
        # If fewer entries than the observable's centrality, skip
        if centrality_index > len(block[g_key]) - 1:
            return ("", "")

        dir_name = block[dir_key]
        g_name = block[g_key][centrality_index]
    else:
        dir_name = block[dir_key]
        g_name = block[g_key]

    # finally, we're not actually interested in the "Graph1D_y", so we remove it on return

    return dir_name, g_name.replace("Graph1D_y", "")


def construct_hepdata_v2_histogram_properties_from_hepdata_v1(
    observable_name: str, config: dict[str, Any], collision_system: str
) -> data.HistogramProperties:
    """Translate common histogram properties from HEPData v1 to HistogramProperties (i.e. HEPData v2).

    Support settings:
    - "xtitle"
    - "ytitle", "ytitle_pp", "ytitle_AA"
    - "logy" (which applies to the AA. pp is set via "log_pp")
    - "y_min_pp", "y_max_pp", "y_min_AA", "y_max_AA"

    This should be called repeatedly for pp, the AA ratio ("AA"), and the AA spectra ("AA_distribution").

    Args:
        observable_name: Observable name.
        config: Observable configuration.
        collision_system: Collision system.
    Returns:
        Histogram properties from the config. Note that this doesn't extract the quantity, since it's not necessarily obvious
        what that value should be.
    """
    axes = {}
    for axis in ["x", "y"]:
        axis_range = (
            config.get(f"{axis}_min_{collision_system}", config.get(f"{axis}_min")),
            config.get(f"{axis}_max_{collision_system}", config.get(f"{axis}_max")),
        )
        log_value = config.get("logy" if collision_system != "pp" else "logy_pp", False)
        axes[axis] = data.Axis(
            label=config.get(f"{axis}title_{collision_system}", config.get(f"{axis}title", "")),
            range=axis_range,
            # We only specified log for the y-axis, so only pass the value in that case!
            log=log_value if axis == "y" else False,
        )

    return data.HistogramProperties(
        # Take the name as the observable name, minus the experiment. This isn't necessarily right, but it's a reasonable start
        quantity="_".join(observable_name.split("_")[:-1]),
        x_axis=axes["x"],
        y_axis=axes["y"],
    )


def is_observable_hepdata_v1(config: dict[str, Any]) -> bool:
    """True if the observable config is HEPData v1.

    NOTE:
        Coming back False doesn't mean that it's necessarily HEPData v2. It could be custom_data, etc.
        It just means that we cannot convert it.

    Args:
        config: Observable config
    Returns:
        True if the block contains a HEPData v1 config.
    """
    return "hepdata" in config or "hepdata_pp" in config or "hepdata_AA" in config


def _encode_param_value(v: Any) -> str:
    """Encode a single parameter value (possibly a ParameterSpec) to a string."""
    if hasattr(v, "encode"):
        return v.encode()
    return str(v)


def _group_entries_by_systematics(
    histograms: list[data.HEPDataEntry],
) -> list[tuple[dict[str, str], dict[str, float], list[data.HEPDataEntry]]]:
    """Group HEPDataEntry objects by systematics only, preserving insertion order.

    Note: table and index are allowed to vary within each group and will be handled
    as part of the combinations.

    Returns:
        List of (systematics_names, additional_systematics_values, entries) tuples.
    """
    # Use a list of keys to preserve insertion order
    group_keys: list[tuple] = []
    group_metadata: dict[tuple, tuple[dict, dict]] = {}
    group_entries: dict[tuple, list[data.HEPDataEntry]] = {}

    for entry in histograms:
        sys_key = tuple(sorted(entry.systematics_names.items()))
        add_sys_key = tuple(sorted(entry.additional_systematics_values.items()))
        key = (sys_key, add_sys_key)

        if key not in group_entries:
            group_keys.append(key)
            group_metadata[key] = (entry.systematics_names, entry.additional_systematics_values)
            group_entries[key] = []

        group_entries[key].append(entry)

    return [
        (sys_names, add_sys, group_entries[k])
        for k, (sys_names, add_sys) in zip(group_keys, [group_metadata[k] for k in group_keys], strict=True)
    ]


def _split_common_and_varying_params(
    entries: list[data.HEPDataEntry],
) -> tuple[dict[str, str], list[str]]:
    """Partition parameter keys into those shared (same encoded value) vs. varying across entries.

    Returns:
        (common_params, varying_keys) where common_params maps key → encoded value
        and varying_keys is the list of keys that differ between entries.
    """
    if not entries:
        return {}, []

    all_keys = list(entries[0].parameters.keys())
    common_params: dict[str, str] = {}
    varying_keys: list[str] = []

    for key in all_keys:
        encoded_values = [_encode_param_value(e.parameters[key]) for e in entries]
        if len(set(encoded_values)) == 1:
            common_params[key] = encoded_values[0]
        else:
            varying_keys.append(key)

    return common_params, varying_keys


def _partition_independent_cartesian_factors(
    entries: list[data.HEPDataEntry],
    varying_keys: list[str],
    enable_factorization: bool = True,
) -> tuple[list[str], list[str]]:
    """Partition varying keys into independent and dependent factors.

    Independent factors: Parameters that form a complete Cartesian product
    across ALL unique (table, index) pairs. When expanded, independent params
    x each combination entry's params must map 1-to-1 to entries.

    Dependent factors: All other varying parameters (those co-varying with
    table/index or not appearing in all (table, index) groups).

    Uses greedy maximization to find the largest independent subset first.

    Args:
        entries: List of HEPDataEntry objects.
        varying_keys: Parameter keys that vary across entries.
        enable_factorization: If False, skip factorization and treat all varying keys as dependent.

    Returns:
        (independent_keys, dependent_keys) where independent_keys are those
        that form a complete Cartesian product across all (table, index) pairs,
        and dependent_keys are all others.
    """
    if not enable_factorization or not varying_keys:
        return [], varying_keys

    # Step 1: Group entries by unique (table, index) pair
    table_index_groups: dict[tuple[str, str], list[data.HEPDataEntry]] = {}
    for entry in entries:
        key = (entry.table, entry.table_index)
        if key not in table_index_groups:
            table_index_groups[key] = []
        table_index_groups[key].append(entry)

    table_index_pairs = list(table_index_groups.keys())

    # Step 2: Try subsets of varying_keys, largest first (greedy maximization)
    for r in range(len(varying_keys), 0, -1):
        for subset in itertools.combinations(varying_keys, r):
            subset_keys = list(subset)
            is_independent = True

            # Check if this subset forms a complete product across ALL (table, index) pairs
            for ti_pair in table_index_pairs:
                pair_entries = table_index_groups[ti_pair]

                # Collect unique values for each key in this subset, within this (table, index) group
                param_values_in_pair: dict[str, list[str]] = {}
                for key in subset_keys:
                    values = {_encode_param_value(e.parameters[key]) for e in pair_entries}
                    param_values_in_pair[key] = sorted(values)

                # Compute expected Cartesian product size for this subset
                expected_size = 1
                for values in param_values_in_pair.values():
                    expected_size *= len(values)

                # For independence: product must equal number of entries in this group
                if expected_size != len(pair_entries):
                    is_independent = False
                    break

                # Verify all combinations actually exist in this group
                expected_combos = set(
                    itertools.product(*[param_values_in_pair[k] for k in subset_keys])
                )
                actual_combos = set(
                    tuple(_encode_param_value(e.parameters[k]) for k in subset_keys)
                    for e in pair_entries
                )

                if expected_combos != actual_combos:
                    is_independent = False
                    break

            # If we found a complete factorization across all groups, return it
            if is_independent:
                dependent = [k for k in varying_keys if k not in subset_keys]
                return subset_keys, dependent

    # No independent factors found; all keys are dependent
    return [], varying_keys


def _build_block_dict(block: data.HEPDataBlock, enable_factorization: bool = True) -> CommentedMap:
    d = CommentedMap()
    # We want the range to show up on a single flow, so we enable the flow style here
    d.update(block.histogram_properties.encode(use_yaml_flow_style=True))

    tables_seq = CommentedSeq()

    for sys_names, add_sys, entries in _group_entries_by_systematics(block.histograms):
        table_entry = CommentedMap()
        common_params, varying_keys = _split_common_and_varying_params(entries)

        # NEW: Partition varying keys into independent vs dependent
        independent_keys, dependent_keys = _partition_independent_cartesian_factors(
            entries, varying_keys, enable_factorization=enable_factorization
        )

        # IMPORTANT: If dependent_keys is empty but table/index vary, we have a problem:
        # We'd need to create combinations with only table/index and no parameters, which is invalid.
        # In this case, treat all varying keys as dependent (fallback to no factorization).
        all_tables = {e.table for e in entries}
        all_indices = {e.table_index for e in entries}
        if not dependent_keys and (len(all_tables) > 1 or len(all_indices) > 1):
            dependent_keys = independent_keys
            independent_keys = []

        # Build outer parameters dict
        outer_params_dict = {**common_params}

        # Add independent factors as lists (or scalars if only one value)
        if independent_keys:
            for key in independent_keys:
                unique_values = sorted(set(_encode_param_value(e.parameters[key]) for e in entries))
                if len(unique_values) > 1:
                    seq = CommentedSeq(unique_values)
                    seq.fa.set_flow_style()
                    outer_params_dict[key] = seq
                else:
                    outer_params_dict[key] = unique_values[0]

        # Set outer parameters if we have any
        if outer_params_dict:
            outer_params = CommentedMap(outer_params_dict)
            outer_params.fa.set_flow_style()
            table_entry["parameters"] = outer_params

        # Systematics (same as before)
        sys_map = CommentedMap(sys_names)
        sys_map.fa.set_flow_style()
        table_entry["systematics_names"] = sys_map

        add_sys_map = CommentedMap(add_sys)
        add_sys_map.fa.set_flow_style()
        table_entry["additional_systematics"] = add_sys_map

        # Put table/index in outer level if they don't vary
        if len(all_tables) == 1:
            table_entry["table"] = entries[0].table
        if len(all_indices) == 1:
            table_entry["index"] = entries[0].table_index

        # Build combinations ONLY if there are dependent parameters
        # (We require at least one parameter per combination entry)
        if dependent_keys:
            combinations = CommentedSeq()
            for entry in entries:
                comb = CommentedMap()

                # Add dependent parameters
                varying_map = CommentedMap(
                    {k: _encode_param_value(entry.parameters[k]) for k in dependent_keys}
                )
                varying_map.fa.set_flow_style()
                comb["parameters"] = varying_map

                # Add table if it varies
                if len(all_tables) > 1:
                    comb["table"] = entry.table

                # Add index if it varies
                if len(all_indices) > 1:
                    comb["index"] = entry.table_index

                combinations.append(comb)

            table_entry["combinations"] = combinations

        tables_seq.append(table_entry)

    d["tables"] = tables_seq
    return d


def write_observable_blocks_to_yaml(
    observable_hep_data: dict[str, data.HEPData],
    output_path: Path | None = None,
    enable_factorization: bool = True,
) -> str:
    """Serialize observable blocks to YAML in the HEPData v2 format.

    Groups HEPDataEntry objects sharing the same table and systematics into a single
    ``tables`` element with a ``combinations`` sub-list.  Parameters that are identical
    across all entries in a group appear once in the outer ``parameters`` dict; the rest
    appear per-combination.

    Args:
        observable_hep_data: Mapping of collision system -> HEPData containing an identifier and observable blocks.
        output_path: If provided, also write the YAML to this file.
        enable_factorization: If True (default), apply greedy factorization to collapse independent
            Cartesian products into compact list form. If False, keep all varying parameters expanded
            in the combinations block.

    Returns:
        YAML string suitable for pasting into an observable config under the ``data:`` key.
    """
    y = ruamel.yaml.YAML()
    y.default_flow_style = False
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120

    # Setup
    # NOTE: The heuristic here is that we only use the custom ruamel.yaml types
    #       if we need to control the appearance of the output. Otherwise, there's
    #       no purpose to using the types. (Plus, they can cause additional issues
    #       since they aren't exactly dicts, so not worth it unless needed).
    root = {"data": {}}

    for collision_system, hep_data in observable_hep_data.items():
        # Build up the "hepdata" map:
        hepdata_map = {
            "record": hep_data.identifier.encode()
        }

        for block_name, block in hep_data.observable_blocks.items():
            hepdata_map[block_name] = _build_block_dict(block, enable_factorization=enable_factorization)

        # Build up the storage structure:
        # "data" -> {collision_system} -> "hepdata"
        root["data"][collision_system] = {
            "hepdata": hepdata_map
        }

    stream = io.StringIO()
    y.dump(root, stream)
    result = stream.getvalue()

    if output_path is not None:
        output_path.write_text(result)
        logger.info(f"Wrote YAML to {output_path}")

    return result


def main(jetscape_analysis_config_path: Path, enable_factorization: bool = True) -> None:
    # We want to update all observables, so let's grab them all
    observables = observable.read_observables_from_config(jetscape_analysis_config_path=jetscape_analysis_config_path)
    # And the data curation database, for convenience
    data_curation_database = hepdata_utils.read_database()

    # NOTE: It's not precisely collision system since we need to grab the AA distribution (i.e. spectra) too,
    #       at least when they're available, so we call it label. It maps to (collision_system, observable_block_name)
    label_to_observable_block = {
        "pp": ("pp", "spectra"),
        "AA_distribution": ("AA", "spectra"),
        "AA": ("AA", "ratio"),
    }

    for obs in observables.values():
        # TEMP: for testing
        #if (obs.sqrt_s, obs.observable_class) != (200, "hadron"):
        #if obs.sqrt_s != 5020 or obs.name != "zg_cms":
        #if obs.sqrt_s != 5020:
        if obs.identifier != (5020, "inclusive_jet", "zg_cms"):
            continue
        # ENDTEMP

        logger.info(f"Processing {obs.identifier}")
        is_v1 = is_observable_hepdata_v1(obs.config)

        if not is_v1:
            # Customize the messages a bit just to help keep track...
            if "data" in obs.config:
                logger.info(f"'{obs.observable_str}' is already HEPData v2")
            elif "custom_data" in obs.config:
                logger.info(f"'{obs.observable_str}' uses custom data and cannot be converted automatically.")
            else:
                logger.info(
                    f"'{obs.observable_str}' is not recognized as HEPData v2 - not sure what this is, so nothing else to be done."
                )
            continue

        # First, let's build the HEPData records, as best we can.
        hepdata_identifiers = {"pp": "", "AA": ""}
        for collision_system in hepdata_identifiers:
            # We'll grab the existing HEPData filename, and then double check with out database
            hepdata_root_filename = obs.config.get(f"hepdata_{collision_system}", obs.config.get("hepdata"))
            if hepdata_root_filename is None:
                logger.warning(f"Could not find hepdata filename. Check the observable: {obs}")
                continue
            _, inspire_hep_id, version, *_ = hepdata_root_filename.split("-")
            inspire_hep_id = int(inspire_hep_id.replace("ins", ""))
            version = int(version.replace("v", ""))
            identifier_from_config = hepdata_utils.HEPDataIdentifier(
                inspire_hep_id=inspire_hep_id,
                version=version,
            )
            # Store identifier for later
            hepdata_identifiers[collision_system] = identifier_from_config

            # And then double check with the data curation database, if the observable is there and that the HEPData matches
            matched_identifiers = [
                v.identifier
                for v in data_curation_database[str(obs.observable_str_as_path)]
                if v.identifier == identifier_from_config
            ]
            if not any(matched_identifiers):
                curated_observable = data_curation_database[
                    str(Path(str(obs.sqrt_s)) / obs.observable_class / obs.name)
                ]
                msg = f"Could not find HEPData identifier for {collision_system=}, {obs}. Double check the database, and possibly add it!"
                msg += "\n" + f"{identifier_from_config=}, {[v.identifier for v in curated_observable]=}"
                msg += "\n" + f"{curated_observable=}"
                logger.warning(msg)
            # else:
            #    logger.warning(f"Found for {obs.name}")

        # Next, extract histogram properties per observable
        histogram_properties_per_observable_block = {
            "pp": {},
            "AA": {},
        }
        for label, (collision_system, observable_block_name) in label_to_observable_block.items():
            histogram_properties_per_observable_block[collision_system][observable_block_name] = (
                construct_hepdata_v2_histogram_properties_from_hepdata_v1(
                    obs.name,
                    obs.config,
                    collision_system=label,
                )
            )

        # Now, we need to retrieve the HEPData information based on the parameters
        parameters = obs.parameters()
        # NOTE: We need the pt binning info solely to generate the older entry name from the parameters
        pt_specs = observable.find_parameter_by_spec_type(parameters, desired_type=observable.PtSpecs)
        n_pt_bins = len(pt_specs.values)

        observable_blocks = {
            "pp": {},
            "AA": {},
        }
        for label, (collision_system, observable_block_name) in label_to_observable_block.items():
            histograms = []
            for params, param_indices in obs.generate_parameter_combinations(parameters=parameters):
                labeled_indices = {f"{k}_index": v for k, v in param_indices.items()}
                suffix, pt_suffix = generate_entry_name_from_parameters(
                    **{**params, **labeled_indices},
                    n_pt_bins=n_pt_bins,
                )
                hepdata_table, hepdata_table_index = find_hepdata_v1_key_in_block(
                    obs.config,
                    # NOTE: We intentionally pass the label here since we need e.g. "AA_distribution" when appropriate.
                    collision_system=label,
                    centrality_index=param_indices["centrality"],
                    suffix=suffix,
                    pt_suffix=pt_suffix,
                )
                histograms.append(
                    data.HEPDataEntry(
                        parameters=params,
                        table=hepdata_table,
                        table_index=hepdata_table_index,
                        # These values are not stored here, so there's nothing to add...
                        # We'll have to fill them in by hand later.
                        systematics_names={},
                        additional_systematics_values={},
                    )
                )
            # And then construct all of the histograms we've seen into a single observable block
            observable_blocks[collision_system][observable_block_name] = data.HEPDataBlock(
                histogram_properties=histogram_properties_per_observable_block[collision_system][observable_block_name],
                histograms=histograms,
            )

        observable_hep_data = {
            "pp": data.HEPData(
                identifier=hepdata_identifiers["pp"],
                observable_blocks=observable_blocks["pp"],
            ),
            "AA": data.HEPData(
                identifier=hepdata_identifiers["AA"],
                observable_blocks=observable_blocks["AA"],
            ),
        }

        # Write the observable blocks to YAML in the HEPData v2 format.
        yaml_str = write_observable_blocks_to_yaml(
            observable_hep_data=observable_hep_data,
            enable_factorization=enable_factorization,
        )
        logger.info(f"\n# --- {obs.observable_str} ---\n{yaml_str}")


if __name__ == "__main__":
    from jetscape_analysis.base import helpers

    helpers.setup_logging(level=logging.DEBUG)

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Convert JETSCAPE-analysis YAML configuration files to a list for PR purposes."
    )
    parser.add_argument(
        "-c",
        "--jetscape-analysis-config",
        type=Path,
        help="Path to the jetscape-analysis config directory. e.g. `config/`",
        required=True,
    )
    parser.add_argument(
        "--no-factorization",
        action="store_true",
        help="Disable greedy factorization of independent Cartesian parameters. "
             "When set, all varying parameters are kept expanded in the combinations block.",
    )
    args = parser.parse_args()

    main(
        jetscape_analysis_config_path=args.jetscape_analysis_config,
        enable_factorization=not args.no_factorization,
    )
