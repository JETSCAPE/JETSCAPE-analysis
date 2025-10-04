from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

from jetscape_analysis.data_curation import hepdata_utils

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
        systematic_names: Map of str -> str from the entry in the HEPdata to what it actually
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
    systematic_names: dict[str, str]
    additional_systematic_values: dict[str, float]


def parse_nested_hepdata_yaml(
    tables: dict[str, Any], expected_parameters: list[str], specified_parameters: dict[str, Any]
) -> ...:
    if "table" in tables and "index" in tables:
        parameters = specified_parameters.copy()
        parameters.update(tables["parameters"])
        # We've found the specification for a histogram.
        # We need to construct the object and include this.
        HEPDataEntry(
            parameters=parameters,
            table=tables["table"],
            table_index=tables["index"],
            # TODO(RJE): Need to somehow have this available...? Presumably just pass this in.
            systematic_names=systematic_names,
            additional_systematic_values=additional_systematic_values,
        )


@attrs.define
class HEPDataBlock:
    """Class of HEPData histograms.

    This contains a list of histograms.

    Attributes:
    """

    histogram_properties: HistogramProperties
    histograms: list[HEPDataEntry]
    # systematic_names: dict[str, str] = attrs.field(factory=dict)
    # additional_systematic_values: dict[str, float] = attrs.field(factory=dict)

    @classmethod
    def from_config(cls, config: Config, expected_parameters: list[str]) -> HEPDataBlock:
        # TODO(RJE): Not sure about the type of expected_parameters...
        histograms = []
        tables = config["tables"]

        parse_nested_hepdata_yaml()

        return cls(
            histogram_properties=HistogramProperties.from_config(config),
        )


@attrs.define
class HEPData:
    """Block of HEPData information corresponding to a set of histogram classes."""

    record: hepdata_utils.HEPDataIdentifier
    additional_params: dict[str, Any] = attrs.Factory(dict)
    # These are the required blocks
    blocks: dict[str, HEPDataBlock]

    @classmethod
    def from_config(cls, collision_system: str, config: Config) -> HEPData:
        # Validation
        _check_for_required_histogram_blocks(collision_system=collision_system, config=config)

        # Retrieve identifier
        identifier = hepdata_utils.HEPDataIdentifier.from_hepdata_config(config)

        keys = list(config.keys())
        # Skip the record since it's the only entry that isn't a histogram block.
        keys.pop("record")

        blocks = {}
        for k in keys:
            blocks[k] = HEPDataBlock.from_config(config[k])

        return cls(
            identifier=identifier,
            additional_params=config["record"].get("additional_params", {}),
            blocks=blocks,
        )


def parse_data_block(observable_str: str, config: dict[str, Any]) -> ...:
    data = config.get("data")

    if not data:
        msg = f"Data block is required for observable: {observable_str}"
        raise ValueError(msg)

    # Handle each collision system
    for collision_system in ["pp", "AA"]:
        collision_system_data = data.get(collision_system)
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
            c = parse_binning_block(observable_str, collision_system_data["binning"])


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


def load_first_production_hepdata_info(observable: observable.Observable) -> HEPDataBlock: ...
