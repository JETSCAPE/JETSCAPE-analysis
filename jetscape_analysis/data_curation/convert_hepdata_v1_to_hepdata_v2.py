"""Convert the HEPdata format from the first production to the new (2026) format

This is mostly a one-off script - we should only need to do the conversion once.
But it should make it less work for us to switch to the new format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import io
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
        logger.info(f"hepdata_{collision_system}_dir{suffix}, {pt_suffix=} not found!")
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


def _group_entries_by_table_and_systematics(
    histograms: list[data.HEPDataEntry],
) -> list[tuple[str, dict[str, str], dict[str, float], list[data.HEPDataEntry]]]:
    """Group HEPDataEntry objects by table + systematics, preserving insertion order.

    Returns:
        List of (table, systematics_names, additional_systematics_values, entries) tuples.
    """
    # Use a list of (key, metadata, entries) to preserve insertion order
    group_keys: list[tuple] = []
    group_metadata: dict[tuple, tuple[str, dict, dict]] = {}
    group_entries: dict[tuple, list[data.HEPDataEntry]] = {}

    for entry in histograms:
        sys_key = tuple(sorted(entry.systematics_names.items()))
        add_sys_key = tuple(sorted(entry.additional_systematics_values.items()))
        key = (entry.table, sys_key, add_sys_key)

        if key not in group_entries:
            group_keys.append(key)
            group_metadata[key] = (entry.table, entry.systematics_names, entry.additional_systematics_values)
            group_entries[key] = []

        group_entries[key].append(entry)

    return [
        (table, sys_names, add_sys, group_entries[k])
        for k, (table, sys_names, add_sys) in zip(group_keys, [group_metadata[k] for k in group_keys], strict=True)
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


def _build_axis_dict(axis: data.Axis) -> CommentedMap:
    d: CommentedMap = CommentedMap()
    d["label"] = axis.label
    if axis.range != (None, None):
        seq: CommentedSeq = CommentedSeq(list(axis.range))
        seq.fa.set_flow_style()
        d["range"] = seq
    if axis.log:
        d["log"] = axis.log
    return d


def _build_block_dict(block: data.HEPDataBlock) -> CommentedMap:
    d: CommentedMap = CommentedMap()
    hp = block.histogram_properties
    d["quantity"] = hp.quantity
    d["x_axis"] = _build_axis_dict(hp.x_axis)
    d["y_axis"] = _build_axis_dict(hp.y_axis)

    tables_seq: CommentedSeq = CommentedSeq()

    for table, sys_names, add_sys, entries in _group_entries_by_table_and_systematics(block.histograms):
        table_entry: CommentedMap = CommentedMap()
        common_params, varying_keys = _split_common_and_varying_params(entries)

        if common_params:
            outer_params: CommentedMap = CommentedMap(common_params)
            outer_params.fa.set_flow_style()
            table_entry["parameters"] = outer_params

        sys_map: CommentedMap = CommentedMap(sys_names)
        sys_map.fa.set_flow_style()
        table_entry["systematics_names"] = sys_map

        add_sys_map: CommentedMap = CommentedMap(add_sys)
        add_sys_map.fa.set_flow_style()
        table_entry["additional_systematics"] = add_sys_map

        table_entry["table"] = table

        if len(entries) == 1 and not varying_keys:
            # Single entry with no variation → no combinations block needed
            table_entry["index"] = entries[0].table_index
        else:
            combinations: CommentedSeq = CommentedSeq()
            for entry in entries:
                comb: CommentedMap = CommentedMap()
                if varying_keys:
                    varying_map: CommentedMap = CommentedMap(
                        {k: _encode_param_value(entry.parameters[k]) for k in varying_keys}
                    )
                    varying_map.fa.set_flow_style()
                    comb["parameters"] = varying_map
                comb["index"] = entry.table_index
                combinations.append(comb)
            table_entry["combinations"] = combinations

        tables_seq.append(table_entry)

    d["tables"] = tables_seq
    return d


def write_observable_blocks_to_yaml(
    observable_blocks: dict[str, dict[str, data.HEPDataBlock]],
    hepdata_identifiers: dict[str, hepdata_utils.HEPDataIdentifier],
    output_path: Path | None = None,
) -> str:
    """Serialize observable blocks to YAML in the HEPData v2 format.

    Groups HEPDataEntry objects sharing the same table and systematics into a single
    ``tables`` element with a ``combinations`` sub-list.  Parameters that are identical
    across all entries in a group appear once in the outer ``parameters`` dict; the rest
    appear per-combination.

    Args:
        observable_blocks: Mapping of collision system (``"pp"``, ``"AA"``) to a dict of
            block name (``"spectra"``, ``"ratio"``) to :class:`~data.HEPDataBlock`.
        hepdata_identifiers: Mapping of collision system to :class:`~hepdata_utils.HEPDataIdentifier`.
        output_path: If provided, also write the YAML to this file.

    Returns:
        YAML string suitable for pasting into an observable config under the ``data:`` key.
    """
    y = ruamel.yaml.YAML()
    y.default_flow_style = False
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120

    root: CommentedMap = CommentedMap()
    data_map: CommentedMap = CommentedMap()
    root["data"] = data_map

    for cs, blocks in observable_blocks.items():
        cs_map: CommentedMap = CommentedMap()
        data_map[cs] = cs_map

        hepdata_map: CommentedMap = CommentedMap()
        cs_map["hepdata"] = hepdata_map

        identifier = hepdata_identifiers.get(cs)
        if identifier:
            rec: CommentedMap = CommentedMap()
            rec["inspire_id"] = identifier.inspire_hep_id
            rec["version"] = identifier.version
            hepdata_map["record"] = rec

        for block_name, block in blocks.items():
            hepdata_map[block_name] = _build_block_dict(block)

    stream = io.StringIO()
    y.dump(root, stream)
    result = stream.getvalue()

    if output_path is not None:
        output_path.write_text(result)
        logger.info(f"Wrote YAML to {output_path}")

    return result


def main(jetscape_analysis_config_path: Path) -> None:
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
        if obs.identifier != (5020, "inclusive_chjet", "axis_alice"):
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

        # Write the observable blocks to YAML in the HEPData v2 format.
        yaml_str = write_observable_blocks_to_yaml(
            observable_blocks=observable_blocks,
            hepdata_identifiers=hepdata_identifiers,
        )
        logger.info(f"\n# --- {obs.observable_str} ---\n{yaml_str}")

        import IPython  # noqa: PLC0415

        IPython.embed()


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
    args = parser.parse_args()

    main(jetscape_analysis_config_path=args.jetscape_analysis_config)
