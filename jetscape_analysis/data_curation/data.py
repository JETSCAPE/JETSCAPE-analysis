from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@attrs.define
class Binning:
    """Custom binning for comparing with experiment."""

    quantity: str
    bins: npt.NDArray[np.float64] = attrs.field(converter=np.asarray)


@attrs.define
class CustomData:
    """Custom data for comparing with experiment.

    Usually this just consists of binning. It needs to be handled
    on a case-by-base basis.
    """

    quantity: str
    filename: Path


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

    parameters: dict[str, Any]
    table: str
    table_index: int
    systematic_names: dict[str, str]
    additional_systematic_values: dict[str, float]


@attrs.define
class HEPDataHistogramClass:
    """Class of HEPData histograms.

    This contains a list of histograms.

    Attributes:
    """

    quantity: str
    y_axis_label: str
    y_axis_range: tuple[float | None, float | None] = attrs.field(factory=lambda: (None, None))
    histograms: list[HEPDataEntry]
    systematic_names: dict[str, str] = attrs.field(factory=dict)
    additional_systematic_values: dict[str, float] = attrs.field(factory=dict)


@attrs.define
class HEPDataBlock:
    """Block of HEPData information corresponding to a set of histogram classes."""

    filename: Path
    histograms: dict[str, HEPDataHistogramClass]
