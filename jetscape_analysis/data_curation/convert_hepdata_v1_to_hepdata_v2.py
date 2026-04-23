"""Convert the HEPdata format from the first production to the new (2026) format

This is mostly a one-off script - we should only need to do the conversion once.
But it should make it less work for us to switch to the new format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import io
import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from jetscape_analysis.data_curation import data, hepdata_utils, observable

logger = logging.getLogger(__name__)


def generate_entry_name_from_parameters(  # noqa: C901
    n_pt_bins: int = 1,
    hadron_pt_index: int | None = None,
    jet_pt_index: int | None = None,
    jet_R: observable.JetRSpec | None = None,
    jet_grooming_settings: observable.GroomingSettingsSpec | None = None,
    jet_axis: observable.JetAxisDifferenceSpec | None = None,
    jet_angularity: observable.AngularitySpec | None = None,
    jet_charge: observable.JetChargeSpec | None = None,
    jet_subjet_R: observable.SubjetRSpec | None = None,
    # kwargs just absorbs the rest of the arguments, so they can be passed blindly
    **kwargs: Any,
) -> tuple[str, str]:
    """Generate HEPData v1 entry name from HEPData v2 parameters.

    We basically dump all of the parameters here and just let it sort it out.

    Args:
        ... Any parameters and indices that we need to build up the HEPData v1 entry name.
        If they're not available, they will be None.
    Returns:
        suffix, pt_suffix to specify the HEPData v1 entry name.
    """
    # Double check to warn the user if there are problems.
    # We expect to process all arguments, except for the indices and selected specs.
    # This list of specs is empirically determined.
    # fmt: off
    expected_specs_to_skip = [
        "centrality",
        "hadron_pt", "hadron_eta", "hadron_trigger_pt", "hadron_trigger_eta",
        "jet_pt", "jet_eta", "jet_eta_R", "jet_rapidity",
        "pion_trigger_pt", "pion_trigger_Et", "pion_trigger_eta", "pion_trigger_smearing",
        "gamma_trigger_pt", "gamma_trigger_Et", "gamma_trigger_eta", "gamma_trigger_isolation", "gamma_trigger_smearing",
        "z_trigger_pt", "z_trigger_mass", "z_trigger_rapidity", "z_trigger_electron_pt", "z_trigger_electron_pt", "z_trigger_electron_eta", "z_trigger_muon_pt", "z_trigger_muon_eta",
    ]
    # fmt: on
    for k, v in kwargs.items():
        if not ("index" in k or any(k == name for name in expected_specs_to_skip)):
            logger.warning(f"Unrecognized argument to generating entry name: {k=}: {v=}")

    # First, start with the pt_suffix since it's simpler.
    # We only want to include if we have multiple pt bins
    # NOTE: This might change in the v2 format, but this is correct for the v1 format.
    pt_suffix = ""
    # We want either the hadron or the jet pt index, but we won't know which one is available a priori
    pt_index = hadron_pt_index if hadron_pt_index is not None else jet_pt_index
    if pt_index is not None and n_pt_bins > 2:
        pt_suffix = f"_pt{pt_index}"

    # Next, onto the full suffix
    suffix = ""
    # jet_R
    if jet_R is not None:
        suffix += f"_R{jet_R.R}"
    # Grooming settings
    # By convention, the subobservable (e.g. axis, ...) goes after the grooming settings if grooming settings are provided.
    # Generated: hepdata_pp_dir_R0.2_zcut0.2_beta0_WTA_SD
    # In config: hepdata_pp_dir_R0.2_zcut0.2_beta0_WTA_SD_pt0
    if jet_grooming_settings:
        if isinstance(jet_grooming_settings.method, observable.SoftDropSpec):
            soft_drop = jet_grooming_settings.method
            # For the alice angularity and mass measurements, we pass through a SoftDrop with z_cut = 0 and beta = 0,
            # so we want to filter out that case. Maybe this is a poor design...
            if soft_drop.z_cut > 0.0 or soft_drop.beta > 0.0:
                suffix += f"_zcut{soft_drop.z_cut:g}_beta{soft_drop.beta:g}"
        elif isinstance(jet_grooming_settings.method, observable.DynamicalGroomingSpec):
            dyg = jet_grooming_settings.method
            suffix += f"_a{dyg.a:g}"
    # Jet-axis difference
    if jet_axis:
        if jet_axis.grooming_settings:
            method = jet_axis.grooming_settings.method
            suffix += f"_zcut{method.z_cut:g}_beta{method.beta:g}"
        suffix += f"_{jet_axis.type}"
    # Angularity
    if jet_angularity:
        suffix += f"_alpha_{jet_angularity.alpha}"
    # Charge
    if jet_charge:
        suffix += f"_k{jet_charge.kappa}"
    # Subjet R
    if jet_subjet_R:
        suffix += f"_r{jet_subjet_R.r}"

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
        # logger.warning(f"{block[g_key]=}")
        g_name = block[g_key][centrality_index]
    else:
        dir_name = block[dir_key]
        g_name = block[g_key]

    # finally, we're not actually interested in the "Graph1D_y", so we remove it on return

    if isinstance(g_name, list):
        logger.warning(f"{g_name=}")
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


def does_observable_contain_hepdata_v1(config: dict[str, Any]) -> bool:
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


def _encode_param_value(v: observable.Encodable) -> str:
    """Encode a single parameter value (a ParameterSpec) to a string."""
    return v.encode()


def _group_entries_by_systematics_only(
    histograms: list[data.HEPDataEntry],
) -> list[tuple[dict[str, str], dict[str, float], list[data.HEPDataEntry]]]:
    """Group HEPDataEntry objects by systematics only, preserving insertion order.

    Table and table_index are allowed to vary within a group, enabling the combinations
    block to capture table/index variation alongside parameter variation.

    Returns:
        List of (systematics_names, additional_systematics_values, entries) tuples.
    """
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

    A parameter subset P is **independent** if, when entries are grouped by (table, table_index),
    every group has the identical set of P-value combinations, AND that set forms a complete
    Cartesian product within P.  Such parameters can be hoisted to the outer ``parameters`` list.
    All remaining varying parameters are **dependent** and must appear in each combination entry.

    Uses greedy maximization (largest subset first) to find the independent subset.
    Tie-breaking within the same subset size follows the original parameter order.

    Args:
        entries: List of HEPDataEntry objects.
        varying_keys: Parameter keys that vary across entries.
        enable_factorization: If False, skip factorization and treat all varying keys as dependent.

    Returns:
        (independent_keys, dependent_keys).  If enable_factorization is False, returns ([], varying_keys).
    """
    if not enable_factorization or not varying_keys:
        return [], varying_keys

    # Group entries by (table, table_index) pair
    table_index_groups: dict[tuple, list[data.HEPDataEntry]] = defaultdict(list)
    for e in entries:
        table_index_groups[(e.table, e.table_index)].append(e)

    # Try all non-empty subsets, starting with largest (greedy maximization)
    for r in range(len(varying_keys), 0, -1):
        for subset in itertools.combinations(varying_keys, r):
            subset_keys = list(subset)

            # For each (table, index) group, compute the frozen set of param-value tuples
            group_combo_sets: list[frozenset] = []
            for group_entries in table_index_groups.values():
                combos = frozenset(
                    tuple(_encode_param_value(e.parameters[k]) for k in subset_keys) for e in group_entries
                )
                group_combo_sets.append(combos)

            # All (table, index) groups must have identical param-combo sets
            if len({frozenset(s) for s in group_combo_sets}) != 1:
                continue

            common_combos = group_combo_sets[0]

            # The common combo set must equal the full Cartesian product of per-key unique values
            param_value_sets = {k: sorted({_encode_param_value(e.parameters[k]) for e in entries}) for k in subset_keys}
            expected = set(itertools.product(*[param_value_sets[k] for k in subset_keys]))
            if common_combos == expected:
                dependent = [k for k in varying_keys if k not in subset_keys]
                return subset_keys, dependent

    # No independent factors found; all keys are dependent
    return [], varying_keys


def _build_combinations(
    entries: list[data.HEPDataEntry],
    dependent_keys: list[str],
    table_varies: bool,
    index_varies: bool,
) -> CommentedSeq:
    """Build the combinations CommentedSeq for a single table block.

    Each unique (dependent_param_values, table, table_index) tuple produces one combination entry.
    The ``parameters`` key is omitted when there are no dependent parameters.
    The ``table`` key is omitted when it is constant across all entries.
    The ``index`` key is omitted when it is constant across all entries.
    """
    seen: dict[tuple, None] = {}
    for e in entries:
        dep_values = tuple(_encode_param_value(e.parameters[k]) for k in dependent_keys)
        seen[(dep_values, e.table, e.table_index)] = None

    combinations = CommentedSeq()
    for dep_values, table, index in seen:
        comb = CommentedMap()
        if dependent_keys:
            varying_map = CommentedMap(dict(zip(dependent_keys, dep_values, strict=True)))
            varying_map.fa.set_flow_style()
            comb["parameters"] = varying_map
        if table_varies:
            comb["table"] = table
        if index_varies:
            comb["index"] = index
        combinations.append(comb)
    return combinations


def _build_block_dict(block: data.HEPDataBlock, enable_factorization: bool = True) -> CommentedMap:
    d = CommentedMap()
    # We want the range to show up on a single flow, so we enable the flow style here
    d.update(block.histogram_properties.encode(use_yaml_flow_style=True))

    tables_seq = CommentedSeq()

    for sys_names, add_sys, entries in _group_entries_by_systematics_only(block.histograms):
        table_entry = CommentedMap()
        common_params, varying_keys = _split_common_and_varying_params(entries)

        # Partition varying keys into independent (can be hoisted to outer list) vs dependent
        independent_keys, dependent_keys = _partition_independent_cartesian_factors(
            entries, varying_keys, enable_factorization=enable_factorization
        )

        # Build outer parameters: constant params + independent params (as lists or scalars)
        outer_params_dict = {**common_params}
        for key in independent_keys:
            unique_values = sorted({_encode_param_value(e.parameters[key]) for e in entries})
            if len(unique_values) > 1:
                seq = CommentedSeq(unique_values)
                seq.fa.set_flow_style()
                outer_params_dict[key] = seq
            else:
                outer_params_dict[key] = unique_values[0]

        if outer_params_dict:
            outer_params = CommentedMap(outer_params_dict)
            outer_params.fa.set_flow_style()
            table_entry["parameters"] = outer_params

        sys_map = CommentedMap(sys_names)
        sys_map.fa.set_flow_style()
        table_entry["systematics_names"] = sys_map

        add_sys_map = CommentedMap(add_sys)
        add_sys_map.fa.set_flow_style()
        table_entry["additional_systematics"] = add_sys_map

        # Determine whether table/index are constant (→ outer level) or varying (→ combinations)
        unique_tables = list(dict.fromkeys(e.table for e in entries))  # preserve order, deduplicate
        unique_indices = list(dict.fromkeys(e.table_index for e in entries))
        table_varies = len(unique_tables) > 1
        index_varies = len(unique_indices) > 1

        if not table_varies:
            table_entry["table"] = unique_tables[0]
        if not index_varies:
            table_entry["index"] = unique_indices[0]

        # Build combinations block when anything varies (dependent params, table, or index)
        if dependent_keys or table_varies or index_varies:
            table_entry["combinations"] = _build_combinations(
                entries, dependent_keys, table_varies=table_varies, index_varies=index_varies
            )

        tables_seq.append(table_entry)

    d["tables"] = tables_seq
    return d


def write_observable_blocks_to_yaml(
    observable_hep_data: dict[str, data.HEPData],
    output_path: Path | None = None,
    enable_factorization: bool = True,
    leading_yaml_keys: list[str] | None = None,
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
        leading_yaml_keys: Keys to include in front of the data fields. Can be useful for e.g.
            making the indentation match with the real YAML file. Default: None, which does not include it.

    Returns:
        YAML string suitable for pasting into an observable config under the ``data:`` key.
    """
    # Validation
    if leading_yaml_keys is None:
        leading_yaml_keys = []

    y = ruamel.yaml.YAML()
    # Use double quotes rather than single quotes
    # y.default_style = '"'
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
        hepdata_map = {"record": hep_data.identifier.encode()}

        for block_name, block in hep_data.observable_blocks.items():
            hepdata_map[block_name] = _build_block_dict(block, enable_factorization=enable_factorization)

        # Build up the storage structure:
        # "data" -> {collision_system} -> "hepdata"
        root["data"][collision_system] = {"hepdata": hepdata_map}

    # Add the leading yaml keys, with the first one corresponding to the outermost level
    for k in reversed(leading_yaml_keys):
        root = {k: root}

    stream = io.StringIO()
    y.dump(root, stream)
    result = stream.getvalue()

    if output_path is not None:
        output_path.write_text(result)
        logger.info(f"Wrote YAML to {output_path}")

    return result


def main(  # noqa: C901
    jetscape_analysis_config_path: Path, write_conversion_to_yaml: bool, enable_factorization: bool = True
) -> None:
    """Convert HEPData v1 entries to HEPData v2"""
    # We want to update all observables, so let's grab them all
    observables = observable.read_observables_from_all_config(
        jetscape_analysis_config_path=jetscape_analysis_config_path
    )
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
        # Option to select particular observables or classes of observables.
        # Super useful for debugging, so I leave them comment out here for when needed
        # if (obs.sqrt_s, obs.observable_class) != (200, "hadron"):
        # if obs.identifier != (5020, "gamma_trigger_jet", "g_cms"):
        # if obs.identifier != (5020, "inclusive_jet", "mg_cms"):
        # if obs.identifier != (5020, "inclusive_jet", "zg_cms"):
        # if obs.identifier != (5020, "inclusive_chjet", "angularity_alice"):
        #     continue

        logger.info(f"Processing {obs.identifier}")
        is_v1 = does_observable_contain_hepdata_v1(obs.config)

        if not is_v1:
            msg = f"'{obs.observable_str}' No HEPData v1 available."
            # Customize the messages a bit just to help keep track...
            if "data" in obs.config:
                msg += " It's already HEPData v2"
            elif "custom_data" in obs.config:
                msg += " It uses custom data and cannot be converted automatically."
            else:
                msg += " It is not recognized as HEPData v2 - not sure what this is, so nothing else to be done."
            logger.info(msg)
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
        # NOTE: We need the pt binning info solely to generate the older entry name from the parameters.
        #       There could be multiple pt specs in the case of a trigger and recoil, so we need to retrieve the correct one.
        try:
            available_pt_specs = observable.find_parameter_by_spec_type(parameters, desired_type=observable.PtSpecs)
            if len(available_pt_specs) > 1:
                # There is more than one, so it's ambiguous. Let's preferentially take the one with the jet or hadron label
                specs_with_jet_or_hadron_label = [
                    s for s in available_pt_specs if "jet" in s.label or "hadron" in s.label
                ]
                pt_specs = specs_with_jet_or_hadron_label[0]

                logger.warning(
                    f"There were multiple pt specs ({available_pt_specs}), so we took the hadron or jet pt spec ({pt_specs}), as that tends to be what we want. You should probably double check this observable..."
                )
            else:
                # Take the pt_spec that's available
                pt_specs = available_pt_specs[0]
            logger.info(f"{pt_specs=}")
            n_pt_bins = len(pt_specs.values)
        except observable.DidNotFindDesiredParameterSpec:
            # There was no pt spec - this basically must be a gamma or Z trigger, which we don't need to convert, so we just skip it.
            logger.debug("Could not find pt_specs for the observable...")
            n_pt_bins = 1

        observable_blocks = {
            "pp": {},
            "AA": {},
        }
        for label, (collision_system, observable_block_name) in label_to_observable_block.items():
            histograms = []
            # We will use these to determine which parameters to write.
            # NOTE: It's import to use the encoded name so we can match the parameter combinations below.
            essential_parameters_labels = [p.encode_name for p in obs.essential_parameters()]
            for params, param_indices in obs.generate_parameter_combinations(parameters=parameters):
                # Encode the index key so we can pass those to the entry_name function and not overwrite the parameters themselves.
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
                        # Only pass the parameters which are essential to the description of the HEPDataEntry
                        # This keeps the parameter list as brief as
                        # NOTE: It's important that we pass the parameters themselves rather than the essential parameters
                        #       since we want ParameterSpec objects here, and the essential parameters are ParameterSpecs.
                        parameters={k: v for k, v in params.items() if k in essential_parameters_labels},
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
        output_path = None
        if write_conversion_to_yaml:
            output_path = Path("config") / "conversion" / f"{obs.observable_str}.yaml"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = write_observable_blocks_to_yaml(
            observable_hep_data=observable_hep_data,
            output_path=output_path,
            enable_factorization=enable_factorization,
            leading_yaml_keys=[obs.observable_class, obs.name],
        )
        if not write_conversion_to_yaml:
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
    parser.add_argument(
        "--write-conversion-to-yaml",
        action="store_true",
        help="Write conversion to dedicated yaml files rather than to stdout."
        "n.b. there is no way to write directly to the config since round tripping becomes problematic",
    )
    args = parser.parse_args()

    main(
        jetscape_analysis_config_path=args.jetscape_analysis_config,
        write_conversion_to_yaml=args.write_conversion_to_yaml,
        enable_factorization=not args.no_factorization,
    )
