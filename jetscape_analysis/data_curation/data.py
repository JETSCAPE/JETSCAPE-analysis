from __future__ import annotations

import copy
import itertools
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

from jetscape_analysis.data_curation import hepdata_utils, observable

logger = logging.getLogger(__name__)

Parameters = dict[str, Any]
Config = dict[str, Any]


@attrs.frozen
class Axis:
    """Axis display configuration.

    Attributes:
        label: Axis label
        range: Display range for the axis. Default: (None, None),
            which corresponds to an automatically set range.
        log: If True, display the axis in log.
    """

    label: str
    range: tuple[float | None, float | None] = attrs.field(factory=lambda: (None, None))
    log: bool = attrs.field(default=False)

    @classmethod
    def from_config(cls, config: Config) -> Axis:
        return cls(
            label=config["label"],
            range=config.get("range", (None, None)),
            log=config.get("log", False),
        )


@attrs.frozen
class HistogramProperties:
    quantity: str
    x_axis: Axis
    y_axis: Axis

    @classmethod
    def from_config(cls, config: Config) -> HistogramProperties:
        return cls(
            quantity=config["quantity"],
            x_axis=config["x_axis"],
            y_axis=config["y_axis"],
        )


@attrs.define
class BinningEntry:
    """Custom binning for comparing with experiment."""

    parameters: Parameters
    bins: npt.NDArray[np.float32 | np.float64] = attrs.field(converter=np.asarray)


@attrs.define
class Binning:
    """Custom binning for comparing with experiment."""

    histogram_properties: HistogramProperties
    histograms: list[BinningEntry]


@attrs.define
class CustomDataEntry:
    """Custom data for a particular set of properties."""

    parameters: Parameters
    bins: npt.NDArray[np.int32 | np.int64 | np.float32 | np.float64] = attrs.field(converter=np.asarray)
    y: npt.NDArray[np.float32 | np.float64] = attrs.field(converter=np.asarray)
    y_err: npt.NDArray[np.float32 | np.float64] = attrs.field(converter=np.asarray)


@attrs.define
class CustomData:
    """Custom data for comparing with experiment.

    Usually this just consists of binning. It needs to be handled
    on a case-by-base basis.
    """

    filename: Path
    histogram_properties: HistogramProperties
    histograms: list[CustomDataEntry]


@attrs.define
class HEPDataEntry:
    """HEPdata (table) entry.

    Attributes:
        parameters: Parameters which are used for this HEPData table entry.
        table: HEPData table identifier. Often of the form "Table N", it could be an
            arbitrary str.
        table_index: HEPData table index. There are sometimes more than one array of
            values in a HEPData table. If so, the index corresponding to the data of interest
            should be specified here. Indexed from 0.
        systematics_names: Map of str -> str from the entry in the HEPdata to what it actually
            corresponds to. If the systematics names are meaningful, this mapping isn't needed.
            However, if they are not, this provides information on how we should interpret each
            systematic.
        additional_systematic_values: Additional systematic values that should be associated with
            the histogram, but for some reason is not available in the table itself. For example,
            global systematics.
    """

    parameters: Parameters
    table: str
    table_index: int
    systematics_names: dict[str, str]
    additional_systematics_values: dict[str, float]

    @classmethod
    def from_flat_config(cls, entry: Config) -> HEPDataEntry:
        """Construct a HEPDataEntry from a dictionary.

        NOTE:
            This requires the parameters to be encoded in the YAML config. It looks somewhat less nice,
            but RJE supposes that it's not the end of the world.

        Args:
            entry: Dict containing parameters, HEPData table and index, and additional systematics.
        Returns:
            HEPDataEntry constructed from the entry.
        """
        return cls(
            parameters={k: observable.SpecDecoderRegistry.decode(k, v) for k, v in entry["parameters"].items()},
            table=entry["table"],
            table_index=entry["index"],
            systematics_names=entry["systematics_names"],
            additional_systematics_values=entry["additional_systematics"],
        )


def _expand_parameters(config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand a single configuration's parameters into all combinations.

    For parameters with list values that contain lists themselves (like centrality),
    each inner list is treated as a single value.

    For parameters with list values that are simple lists (like jet_R),
    each element is a separate value to combine.
    """
    if "parameters" not in config:
        yield config
        return

    parameters = config["parameters"]

    # Separate parameters that need expansion from those that don't
    params_to_expand = {}
    params_fixed = {}

    for key, value in parameters.items():
        if isinstance(value, list) and len(value) > 0:
            # Check if this is a list of values to expand
            # If the first element is a list, treat each sublist as a single value
            if isinstance(value[0], list):
                # e.g., centrality: [[0, 10], [30, 50]]
                params_to_expand[key] = value
            else:
                # e.g., jet_R: [0.2, 0.3]
                params_to_expand[key] = value
        else:
            # Single value, no expansion needed
            params_fixed[key] = value

    # If no parameters to expand, return the config as-is
    if not params_to_expand:
        yield config
        return

    # Generate all combinations
    param_names = list(params_to_expand.keys())
    param_values = [params_to_expand[name] for name in param_names]

    for combination in itertools.product(*param_values):
        # Create new config for this combination
        new_config = copy.deepcopy(config)
        new_params = copy.deepcopy(params_fixed)

        # Add the combined parameters
        for name, value in zip(param_names, combination, strict=True):
            new_params[name] = value

        new_config["parameters"] = new_params
        yield new_config


def _expand_parameter_combinations_into_individual_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a configuration with combinatorial parameters into individual configs.

    Args:
        config: Configuration dictionary that may contain a "combinations" key

    Returns:
        List of expanded configuration dictionaries
    """
    # If there are combinations, process them first (before expanding parameters)
    if config.get("combinations"):
        combinations = config["combinations"]
        base_config = {k: v for k, v in config.items() if k != "combinations"}

        # Recursively expand each combination
        all_results = []
        for combination in combinations:
            # Merge base config with this combination
            merged = _merge_configs(base_config, combination)

            # Recursively expand in case this combination has its own combinations
            expanded = _expand_parameter_combinations_into_individual_configs(merged)
            all_results.extend(expanded)

        return all_results

    # No more combinations - now expand the parameters
    return list(_expand_parameters(config))


def expand_parameter_combinations_into_individual_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand a list of configurations.

    Args:
        configs: List of configuration dictionaries

    Returns:
        List of all expanded configurations
    """
    all_results = []
    for config in configs:
        expanded = _expand_parameter_combinations_into_individual_configs(config)
        all_results.extend(expanded)
    return all_results


def _merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two configurations, with override taking precedence.

    Special handling for the 'parameters' key to merge dictionaries.
    """
    logger.info(f"Calling merging configs with {base=}, {override=}")
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key == "parameters" and key in result:
            # Merge parameters dictionaries
            result["parameters"] = {**result["parameters"], **value}
        else:
            result[key] = copy.deepcopy(value)

    return result


@attrs.define
class HEPDataBlock:
    """Class of HEPData histograms.

    This contains a list of histograms.

    Attributes:
        histogram_properties: Shared properties of the histograms - e.g. axes titles, etc.
        histograms: List of HEPDataEntry objects, with each one corresponding to a
            single set of properties and a HEPData table + index.
    """

    histogram_properties: HistogramProperties
    histograms: list[HEPDataEntry]
    # systematics_names: dict[str, str] = attrs.field(factory=dict)
    # additional_systematics_values: dict[str, float] = attrs.field(factory=dict)

    @classmethod
    def from_config(cls, config: Config) -> HEPDataBlock:
        """Construct the HEPData block from a yaml configuration.

        Args:
            config: Configuration containing the HEPData block. e.g. the values of a "spectra" block.
        Returns:
            Block of HEPData configurations.
        """
        # Histogram configurations are contained in the "tables"
        tables_config = config["tables"]
        # We need to expand the parameters into
        full_set_of_table_configs = expand_parameter_combinations_into_individual_configs(tables_config)

        # And then construct all of the histograms
        # NOTE: This requires the parameters to be encoded in the YAML config. It looks somewhat less nice,
        #       but RJE supposes that it's not the end of the world.
        logger.info(f"{full_set_of_table_configs=}")
        histograms = []
        # There can be multiple histograms
        # NOTE: The list comprehension is cleaner, but for debugging, it can be helpful to split out the list
        # for entry in full_set_of_table_configs:
        #     logger.info(f"{entry=}")
        #     histograms.append(HEPDataEntry.from_flat_config(entry))
        histograms = [HEPDataEntry.from_flat_config(entry) for entry in full_set_of_table_configs]

        return cls(
            histogram_properties=HistogramProperties.from_config(config),
            histograms=histograms,
        )


@attrs.define
class HEPData:
    """Block of HEPData information corresponding to a set of histogram classes."""

    identifier: hepdata_utils.HEPDataIdentifier
    additional_params: dict[str, Any] = attrs.Factory(dict)
    # These are the required blocks
    observable_blocks: dict[str, HEPDataBlock] = attrs.Factory(dict)

    @classmethod
    def from_config(cls, collision_system: str, config: Config) -> HEPData:
        # Validation
        _check_for_required_histogram_blocks(collision_system=collision_system, config=config)

        # Retrieve identifier
        identifier = hepdata_utils.HEPDataIdentifier.from_hepdata_config(config)

        keys = list(config.keys())
        # Skip the record since it's the only entry that isn't a histogram block.
        keys.pop(keys.index("record"))

        observable_blocks = {}
        for k in keys:
            observable_blocks[k] = HEPDataBlock.from_config(config[k])

        return cls(
            identifier=identifier,
            additional_params=config["record"].get("additional_params", {}),
            observable_blocks=observable_blocks,
        )


class MissingDataBlock(ValueError):
    """Missing data block in the observable definition."""


class MissingPPDataBlock(MissingDataBlock):
    """Missing the pp data block in the observable definition."""


class MissingAADataBlock(MissingDataBlock):
    """Missing the AA data block in the observable definition."""


def parse_data_block(observable_str: str, config: dict[str, Any]) -> ...:
    data = config.get("data")

    if not data:
        msg = f"Data block is required for observable: {observable_str}"
        raise MissingDataBlock(msg)

    # Handle each collision system
    data_blocks = {}
    for collision_system in ["pp", "AA"]:
        collision_system_data = data.get(collision_system, {})
        if not collision_system_data:
            logger.info(f"Unable to find {collision_system} data for {observable_str}")

        # Now handle the possible options:
        if "hepdata" in collision_system_data:
            # TODO(RJE): Need to pass in expected parameters. Get the logic
            #            from define_data_sources(?)
            c = HEPData.from_config(collision_system=collision_system, config=collision_system_data["hepdata"])
        elif "custom" in collision_system_data:
            c = parse_custom_data_block(observable_str, collision_system_data["custom"])
        elif "bins" in collision_system_data:
            c = parse_binning_block(observable_str, collision_system_data["bins"])
        else:
            msg = f"Missing data block for '{collision_system}' for '{observable_str}'"
            if collision_system == "pp":
                raise MissingPPDataBlock(msg)
            raise MissingAADataBlock(msg)

        data_blocks[collision_system] = c

    return data_blocks


def parse_custom_data_block(observable_str: str, config: Config) -> CustomData: ...


def parse_binning_block(observable_str: str, config: Config) -> Binning: ...


def _check_for_required_histogram_blocks(collision_system: str, config: Config) -> bool:
    required_hists = {
        "pp": ["spectra"],
        "AA": ["spectra", "ratio"],
    }
    available_keys = list(config.keys())
    required_hists_available = all(h in available_keys for h in required_hists[collision_system])
    if not required_hists_available:
        msg = f"Missing required keys for {collision_system}. Needed: {required_hists[collision_system]}, available: {available_keys}"
        raise ValueError(msg)

    return True


def load_first_production_hepdata_info(observable: observable.Observable) -> HEPDataBlock:
    """Load HEPData info from the first production.

    Our first production had everything flattened out.

    See the other functionality in define_dat_sources(?) where I've already done this...
    """


def example_search_for_table(obs: observable.Observable) -> None:
    """Example function in searching for a table. Adapt as appropriate."""

    # Full example for one observable: construct combinations from the available parameters.
    # TODO(RJE): How would this work for the double_ratio?
    #            The best options seems like:
    #             - Turn off validation for that block and handle by hand in the histogramming.
    #             - Alternatively, would need the double R ratio definition in the parameters. But this could get trickier.
    #
    # Select an observable, and retrieve the tables
    tables = obs.config["data"]["pp"]["hepdata"]["spectra"]["tables"]
    # We could have multiple tables, but in this particular case, we set it up with one entry in the tables,
    # which we need to expand
    res = expand_parameter_combinations_into_individual_configs(tables[0])
    # Then determine what parameters are available based on the observable parameter combinations
    # NOTE: This requires the parameters to be encoded. It looks somewhat less nice, but I suppose it's not the end of the world.
    available_parameters = [
        {k: observable.SpecDecoderRegistry.decode(k, v) for k, v in r["parameters"].items()} for r in res
    ]
    g = obs.generate_parameter_combinations(obs.parameters())
    # Grab the first one
    desired_parameters = next(g)
    # And find it
    index_of_table_corresponding_to_desired_parameters = available_parameters.index(desired_parameters[0])
    logger.warning(f"{index_of_table_corresponding_to_desired_parameters=}")

    import IPython  # noqa: PLC0415

    IPython.embed()


def example_construct_observables(observables: dict[str, observable.Observable]) -> None:
    """Example of constructing a data block for an observable."""

    data_blocks = {}
    for observable_str, obs in observables.items():
        # Skip for testing...
        if "ktg" not in observable_str:
            continue
        data_blocks[observable_str] = parse_data_block(observable_str=observable_str, config=obs.config)

    import IPython  # noqa: PLC0415

    IPython.embed()


def build_hepdata_repository(observables: dict[str, observable.Observable]) -> None:
    """Retrieve all available data from HEPData and store it in our repository."""
    import time  # noqa: PLC0415

    from jetscape_analysis.base import helpers  # noqa: PLC0415

    helpers.setup_logging(level=logging.INFO)

    for observable_str, obs in observables.items():
        logger.info(f"Processing '{observable_str}'")

        # Attempt to use the new data block, but then fall back to the URLs if not provided
        # (the URLs are less preferred because there could be multiple HEPdata for one observable).
        data_block = {}
        try:
            data_block = parse_data_block(observable_str=observable_str, config=obs.config)
        except MissingDataBlock:
            logger.info("Missing data block - falling back to urls")
        if not data_block:
            urls = obs.config.get("urls", {})
            hepdata_url = urls.get("hepdata")
            if not hepdata_url or hepdata_url == "N/A":
                logger.info(f"Skipping {observable_str} due to missing HEPData url")
                continue

        inspire_hep_id, version, query_params = hepdata_utils.extract_info_from_hepdata_url(
            obs.config["urls"]["hepdata"]
        )

        observable_str_as_path = Path(str(obs.sqrt_s)) / obs.observable_class / obs.name
        _info = hepdata_utils.retrieve_observable_hepdata(
            observable_str_as_path=observable_str_as_path, inspire_hep_id=inspire_hep_id, version=version
        )

        logger.info(f"Completed '{observable_str}'")

        # We don't want to hit HEPData too often, so we slow it down a bit
        time.sleep(0.5)


def main() -> None:
    """Testing function"""

    from jetscape_analysis.base import helpers  # noqa: PLC0415

    helpers.setup_logging(level=logging.INFO)

    observables = observable.read_observables_from_config(jetscape_analysis_config_path=Path("config"))

    # example_search_for_table(obs=observables["5020_inclusive_chjet_ktg_alice"])
    # example_construct_observables(observables=observables)
    build_hepdata_repository(observables=observables)
    # sorted_obs = dict(sorted(observables.items(), key=lambda o: (o[1].observable_class, o[1].sqrt_s, o[1].name)))
    # output = {}
    # for k, obs in sorted_obs.items():
    #     if obs.observable_class not in output:
    #         output[obs.observable_class] = []
    #     s = ", ".join(map(str, (obs.sqrt_s, obs.experiment, obs.name, obs.display_name)))
    #     output[obs.observable_class].append(s)

    # for k, v in output.items():
    #     logger.info(f"\t{k}")
    #     for a in v:
    #         logger.info(a)
    #         #logger.info(", ".join(a))


if __name__ == "__main__":
    main()
