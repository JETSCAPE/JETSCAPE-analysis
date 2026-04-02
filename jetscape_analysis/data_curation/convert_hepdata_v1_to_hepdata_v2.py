"""Convert the HEPdata format from the first production to the new (2026) format

This is mostly a one-off script - we should only need to do the conversion once.
But it should make it less work for us to switch to the new format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from typing import Any

from jetscape_analysis.data_curation import data, hepdata_utils, observable

logger = logging.getLogger(__name__)


def generate_entry_name_from_parameters(
    jet_R: observable.JetRSpec | None = None,
    pt_index: int | None = None,
    kappa: observable.AngularitySpec | None = None,
    soft_drop: observable.SoftDropSpec | None = None,
    n_pt_bins: int = 1,
    **kwargs: Any,
) -> str:
    """Generate HEPData v1 entry name from HEPData v2 parameter specifications."""

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
    if soft_drop is not None:
        suffix += f"_zcut{soft_drop.z_cut}_beta{soft_drop.beta}"
    # kappa
    if kappa is not None:
        suffix += f"_k{kappa.kappa}"

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
        dir_key = f"hepdata_collision_{collision_system}_dir{suffix}{pt_suffix}"
    elif f"hepdata_{collision_system}_dir" in block:
        dir_key = f"hepdata_{collision_system}_dir"
    else:
        logger.info(f"hepdata_{collision_system}_dir{suffix} not found!")
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

    return dir_name, g_name.replace("Graph1D_y")


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
            _, inspire_id, version, *_ = hepdata_root_filename.split("-")
            inspire_id = int(inspire_id.replace("ins", ""))
            version = int(version.replace("v", ""))
            identifier_from_config = hepdata_utils.HEPDataIdentifier(
                inspire_hep_id=inspire_id,
                version=version,
            )
            # Store for later...
            hepdata_identifiers[collision_system] = identifier_from_config

            # And then double check with the data curation database, if the observable is there...
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

            continue

    if False:
        # Extract histogram properties per observable
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
        centrality_specs = observable.find_parameter_by_spec_type(parameters, desired_type=observable.CentralitySpecs)
        n_centrality_bins = len(centrality_specs.values)
        pt_specs = observable.find_parameter_by_spec_type(parameters, desired_type=observable.CentralitySpecs)
        n_pt_bins = len(pt_specs.values)
        # TODO(RJE): I think I don't even need this - I can get the enumerate values from the parameter generator, no?

        observable_blocks = {
            "pp": {},
            "AA": {},
        }
        for label, (collision_system, observable_block_name) in label_to_observable_block.items():
            histograms = []
            for i, (params, param_indices) in enumerate(obs.generate_parameter_combinations(parameters=parameters)):
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
                # params_index_to_tables[i] = (params, *hepdata_table_and_index)
                # hepdata_entries[collision_system][observable_block] = data.HEPDataEntry(
                histograms.append(
                    data.HEPDataEntry(
                        parameters=params,
                        table=hepdata_table,
                        table_index=hepdata_table_index,
                        # These are not stored here, so there's nothing to add...
                        systematics_names={},
                        additional_systematics_values={},
                    )
                )
            # And then construct all of the histograms we've seen into a single observable block
            observable_blocks[collision_system][observable_block_name] = data.HEPDataBlock(
                histogram_properties=histogram_properties_per_observable_block[collision_system][observable_block_name],
                histograms=histograms,
            )

        # for collision_system, observable_blocks in hepdata_entries.items():
        #    for observable_block, entry in observable_blocks.items():
        #        observable_blocks[collision_system][observable_block] = data.HEPDataBlock(
        #            histogram_properties=histogram_properties_per_observable_block[collision_system]
        #        )

        import IPython

        IPython.embed()

        raise RuntimeError("")


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
