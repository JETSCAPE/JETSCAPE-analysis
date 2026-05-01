"""Represent the information about a single Observable.

An Observable is measured for a set parameters, such as jet_R, pt, etc.
This module defines how that observable should be treated, and how those
parameters are represented by observables.

For each parameter, we define a `ParameterSpec` (i.e. parameter specification)
which represents a single value of that parameter. Each specification contains
the information needed to define that parameter (could be a single value,
such as jet R, or could be multiple values, such as in the centrality range),
and serialization information (more on this below).

A set of parameter values are stored in a `ParameterSpecs` class (note the plural
name!). This contains the ParameterSpec values of the same type (e.g. JetRSpecs
contains a JetRSpec per value of jet R that is measured in the analysis). It also
includes the names (internal + display), and allows serialization of the full set of
values. The names are stored with the Specs rather than with an individual Spec because
the Spec corresponds to a single value we can use, independent of where it's from!
The Specs object then locates it - e.g. with a label.

## Serialization

Each `ParameterSpec` and `ParameterSpecs` class are defined to be serialized
in and out of the yaml configuration file. The concept here is to enforce
consistency on how parameters are defined.

- Serialization for individual specifications is handled through the `SpecDecoderRegistry`
- Serialization for set of parameter specifications is handled through the `SpecsDecoderRegistry`

In each case, they are expected to implement the `Encodable` protocol.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

import attrs
import numpy as np
import yaml

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@runtime_checkable
class Encodable(Protocol):
    """Interface for an object that can be encoded and decoded."""

    def encode(self) -> str: ...
    @classmethod
    def decode(cls, value: str) -> Encodable: ...


EncodableT = TypeVar("EncodableT", bound=Encodable)


class DecoderRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[Encodable]] = {}

    def register(self, name: str, decoder_type: type[Encodable]) -> None:
        self._registry[name] = decoder_type

    def get_type(self, name: str) -> type[Encodable]:
        # If the name is immediately in the registry, just use it.
        if name in self._registry:
            return self._registry[name]
        # Handle labeled names produced by ParameterSpecs.encode_name where we have f"{label}_{spec_name}".
        # The strategy is to try the longest suffix first, and then narrow down if nothing is found.
        # This should ensure that e.g. "eta_R" is found rather than "R".
        parts = name.split("_")
        for i in range(1, len(parts)):
            suffix = "_".join(parts[i:])
            if suffix in self._registry:
                return self._registry[suffix]
        msg = f"Unknown decoder type: {name}"
        raise ValueError(msg)

    def get_name(self, instance: Encodable) -> str:
        for k, v in self._registry.items():
            if v is type(instance):
                return k
        msg = f"Unable to find name corresponding to type {type(instance)}"
        raise ValueError(msg)

    def decode(self, name: str, value: str) -> Encodable:
        decoder_type = self.get_type(name)
        return decoder_type.decode(value)

    def decode_as(self, expected_type: type[EncodableT], name: str, value: str) -> EncodableT:
        decoder_type = self.get_type(name)
        if decoder_type != expected_type:
            msg = f"Expected {expected_type.__name__}, but {name} is registered as {decoder_type.__name__}"
            raise TypeError(msg)
        # NOTE: The cast is necessary since mypy won't be able to type infer through all of this generic code
        return cast(EncodableT, decoder_type.decode(value))


class ParameterSpec:
    """Defines a single parameter specification.

    This is just "Encodable", but the more specific name is helpful
    for designing the rest of the code.
    """

    def encode(self) -> str:
        raise NotImplementedError

    @classmethod
    def decode(cls, value: str) -> ParameterSpec:
        raise NotImplementedError


# Decoder registry for a single parameter
SpecDecoderRegistry = DecoderRegistry()


@attrs.frozen
class CentralitySpec(ParameterSpec):
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.low}-{self.high}%"

    def encode(self) -> str:
        return f"{self.low!s}_{self.high!s}"

    @classmethod
    def decode(cls, value: str) -> CentralitySpec:
        # `value` is of the form: "{self.low}_{self.high}"
        # indices:                 0           1
        low, high = value.split("_")
        return cls(low=float(low), high=float(high))


@attrs.frozen
class PtSpec(ParameterSpec):
    low: float
    high: float | None

    def __str__(self) -> str:
        if self.high is None:
            return f"pt >= {self.low}"
        return f"{self.low} <= pt < {self.high}"

    def encode(self) -> str:
        high = -1 if self.high is None else self.high
        return f"{self.low!s}_{high!s}"

    @classmethod
    def decode(cls, value: str) -> PtSpec:
        # `value` is of the form: "{self.low}_{self.high}"
        # indices:                 0           1
        low, high = value.split("_")
        return cls(low=float(low), high=float(high) if not np.isclose(float(high), -1.0) else None)


@attrs.frozen
class EtSpec(ParameterSpec):
    low: float
    high: float | None

    def __str__(self) -> str:
        if self.high is None:
            return f"Et >= {self.low}"
        return f"{self.low} <= Et < {self.high}"

    def encode(self) -> str:
        high = -1 if self.high is None else self.high
        return f"{self.low!s}_{high!s}"

    @classmethod
    def decode(cls, value: str) -> EtSpec:
        # `value` is of the form: "{self.low}_{self.high}"
        # indices:                 0           1
        logger.warning(f"{value=}")
        low, high = value.split("_")
        return cls(low=float(low), high=float(high) if not np.isclose(float(high), -1.0) else None)


@attrs.frozen
class EtaSpec(ParameterSpec):
    """Eta Specification.

    NOTE:
        This is handled differently than other specs that allow ranges since we usually specify only
        a max value for eta. It would be a pain to have to use kwargs
    """

    max: float
    min: float | None = attrs.field(default=None, kw_only=True)

    def __str__(self) -> str:
        if self.min is None:
            return f"|eta|<={self.max}"
        return f"{self.min} <= |eta| < {self.max}"

    def encode(self) -> str:
        if self.min is None:
            return f"{self.max!s}"
        return f"{self.min!s}_{self.max!s}"

    @classmethod
    def decode(cls, value: str) -> EtaSpec:
        # `value` is of the form...
        # Max only: "{self.max}"
        # Min+max : "{self.min}_{self.max}"
        split = value.split("_")
        if len(split) == 1:
            return cls(
                max=float(value),
            )
        return cls(
            min=float(split[0]),
            max=float(split[1]),
        )


@attrs.frozen
class RapiditySpec(ParameterSpec):
    value: float

    def __str__(self) -> str:
        return f"|y|<={self.value}"

    def encode(self) -> str:
        return f"{self.value!s}"

    @classmethod
    def decode(cls, value: str) -> RapiditySpec:
        # `value` is of the form: "{self.value}"
        return cls(
            value=float(value),
        )


@attrs.frozen
class MassSpec(ParameterSpec):
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.low} <= mass < {self.high}"

    def encode(self) -> str:
        return f"{self.low!s}_{self.high!s}"

    @classmethod
    def decode(cls, value: str) -> MassSpec:
        # `value` is of the form: "{self.low}_{self.high}"
        # indices:                  0          1
        split = value.split("_")
        return cls(
            low=float(split[0]),
            high=float(split[1]),
        )


@attrs.frozen
class JetRSpec(ParameterSpec):
    R: float

    def __str__(self) -> str:
        return f"Jet_R={self.R}"

    def encode(self) -> str:
        return f"{self.R!s}"

    @classmethod
    def decode(cls, value: str) -> JetRSpec:
        # `value` is of the form: "{self.R}"
        return cls(
            R=float(value),
        )


@attrs.frozen
class SoftDropSpec(ParameterSpec):
    """Soft drop spec.

    NOTE:
        This isn't used directly in Specs, but rather through the GroomingSettingsSpec, since we need to
        be able to mix the specification of grooming methods.
    """

    z_cut: float
    beta: float

    def __str__(self) -> str:
        return f"Soft Drop(z_cut={self.z_cut}, beta={self.beta})"

    def encode(self) -> str:
        # By convention, we have 0.2, but we want 020, so we multiply by 100
        return f"z_cut_{int(self.z_cut * 100):03g}_beta_{self.beta}"

    @classmethod
    def decode(cls, value: str) -> SoftDropSpec:
        # `value` is of the form: "z_cut_{self.z_cut}_beta_{self.beta}"
        # indices:                 0 1    2           3     4
        split = value.split("_")
        return cls(
            # By convention, we write 020, but we want 0.2, so we divide by 100
            z_cut=float(split[2]) / 100,
            beta=float(split[4]),
        )

    def to_dict(self) -> dict[str, str | float]:
        """Convert the type to a dictionary.

        Only for use in the analysis code, where it makes comparisons much easier!

        Args:
            None
        Returns:
            Dictionary with the Soft Drop attributes.
        """
        return {
            "type": "soft_drop",
            "z_cut": self.z_cut,
            "beta": self.beta,
        }


@attrs.frozen
class DynamicalGroomingSpec(ParameterSpec):
    """Dynamical grooming spec.

    NOTE:
        This isn't used directly in Specs, but rather through the GroomingSettingsSpec, since we need to
        be able to mix the specification of grooming methods.
    """

    a: float

    def __str__(self) -> str:
        return f"Dynamical Grooming(a={self.a})"

    def encode(self) -> str:
        return f"a_{self.a}"

    @classmethod
    def decode(cls, value: str) -> DynamicalGroomingSpec:
        # `value` is of the form: "a_{self.a}"
        # indices:                 0  1
        split = value.split("_")
        return cls(
            a=float(split[1]),
        )

    def to_dict(self) -> dict[str, str | float]:
        """Convert the type to a dictionary.

        Only for use in the analysis code, where it makes comparisons much easier!

        Args:
            None
        Returns:
            Dictionary with the Dynamical Grooming attributes.
        """
        return {
            "type": "dynamical_grooming",
            "a": self.a,
        }


def convert_to_grooming_method_spec(
    value: SoftDropSpec | DynamicalGroomingSpec | dict[str, float | str],
) -> SoftDropSpec | DynamicalGroomingSpec:
    """Convert a possible grooming method spec or arguments to a confirmed grooming method spec."""
    # If it's already a grooming spec or None, nothing else to be done
    if isinstance(value, SoftDropSpec | DynamicalGroomingSpec):
        return value
    # Now, check for dict arguments for different grooming methods
    # We'll determine the spec from the type
    mutable_value = value.copy()
    type = mutable_value.pop("type")
    match type:
        case "soft_drop":
            return SoftDropSpec(**mutable_value)
        case "dynamical_grooming":
            return DynamicalGroomingSpec(**mutable_value)
        case _:
            # Call through to the ValueError
            ...

    msg = f"Could not convert grooming_methods argument into specs. Provided: {value=}"
    raise ValueError(msg)


# Convenience dictionaries to aid in conversion back and forth
GROOMING_SETTINGS_TYPE_TO_LABEL: dict[SoftDropSpec | DynamicalGroomingSpec, str] = {
    SoftDropSpec: "SD",
    DynamicalGroomingSpec: "DyG",
}
GROOMING_SETTINGS_LABEL_TO_TYPE: dict[str, SoftDropSpec | DynamicalGroomingSpec] = {
    v: k for k, v in GROOMING_SETTINGS_TYPE_TO_LABEL.items()
}


@attrs.frozen
class GroomingSettingsSpec(ParameterSpec):
    """Grooming settings specification, which contains the settings of a particular grooming method.

    I'm being a little loose with the language, since here "method" refers to the method + some particular settings specification.
    However, it's useful to have some separate label since this is already called a GroomingSettingsSpec

    NOTE:
        This additional level of indirection is necessary since we want to be able to mix SoftDrop
        and Dynamical Grooming together. They're exclusive methods (i.e. we wouldn't apply both at once),
        so we only want to specify one at a time.

    NOTE:
        We support passing arguments to the method as a dictionary. If this is done, it needs to include:
        {"type": name_of_grooming_method, "grooming_settings_1": a, "grooming_setting_2": b}. Accepted types
        are "soft_drop" and "dynamical_grooming".

    Attributes:
        method: Settings for a grooming method.
    """

    # The point with the convert here is that we'll be passed arguments that are a dict with SoftDrop or Dynamical Grooming
    # parameters, so we need to ensure that it's actually of the expected type.
    method: SoftDropSpec | DynamicalGroomingSpec = attrs.field(converter=convert_to_grooming_method_spec)

    def __str__(self) -> str:
        return f"Grooming Settings ({self.method!s}))"

    def encode(self) -> str:
        output = f"{GROOMING_SETTINGS_TYPE_TO_LABEL[type(self.method)]}"
        output += f"_{self.method.encode()}"
        return output

    @classmethod
    def decode(cls, value: str) -> GroomingSettingsSpec:
        # `value` is of the form: "{grooming_type}_{grooming_settings...}"
        # indices:                  0              1
        split = value.split("_")
        grooming_type = GROOMING_SETTINGS_LABEL_TO_TYPE[split[0]]
        grooming_settings = grooming_type.decode("_".join(split[1:]))

        return cls(method=grooming_settings)


def _convert_to_grooming_settings_spec(
    value: GroomingSettingsSpec | SoftDropSpec | DynamicalGroomingSpec | dict[str, float],
) -> SoftDropSpec | DynamicalGroomingSpec:
    """Convert a grooming settings spec, grooming method spec, or arguments, to a GroomingSettingsSpec.

    NOTE:
        We particularly accept the underlying SoftDropSpec or DynamicalGroomingSpec
    """
    # If it's already a grooming settings spec, nothing else to be done
    if isinstance(value, GroomingSettingsSpec):
        return value
    # If it's already the underlying method, we should pass it through to the standard conversion method
    if isinstance(value, SoftDropSpec | DynamicalGroomingSpec):
        return GroomingSettingsSpec(value)
    # In the case of a map of arguments, we need some separate processing to handle the additional "type" argument
    if isinstance(value, Mapping):
        return GroomingSettingsSpec(value)

    msg = f"Could not convert grooming_methods argument into specs. Provided: {value=}"
    raise ValueError(msg)


@attrs.frozen
class JetAxisDifferenceSpec(ParameterSpec):
    """Jet-axis difference specification.

    NOTE:
        Possible axis values include "WTA_Standard", "WTA_SD", and "Standard_SD".

    Args:
        type: Type of the jet-axis difference. Note that each method should be separated by an "_"!
            For consistency, you should not use "-".
    """

    type: str = attrs.field(converter=lambda x: x.replace("-", "_"))
    # The point with the convert here is that we'll be passed arguments that are a dict with SoftDrop or Dynamical Grooming
    # parameters, so we need to ensure that it's actually of the expected type.
    grooming_settings: GroomingSettingsSpec | None = attrs.field(
        default=None, converter=lambda x: _convert_to_grooming_settings_spec(x) if x is not None else None
    )

    def __str__(self) -> str:
        output = f"Jet-axis difference, {self.type.replace('_', '-')}"
        if self.grooming_settings:
            output = f"w/ {self.grooming_settings!s}"
        return output

    def encode(self) -> str:
        output = f"{self.type}"
        if self.grooming_settings:
            output += f"_{self.grooming_settings.encode()}"
        return output

    @classmethod
    def decode(cls, value: str) -> JetAxisDifferenceSpec:
        # `value` is of the form: "{type}_{grooming_settings}"
        # where type is of the form "method1_method2", so the indices are offset by +1
        # e.g.     "method1_method2_{grooming_settings}"
        # indices:  0      (1)       (2) ...
        split = value.split("_")
        grooming_settings = None
        # Grooming settings start at index 2
        if len(split) > 2:
            grooming_settings = GroomingSettingsSpec.decode("_".join(split[2:]))
        return cls(type="_".join(split[:2]), grooming_settings=grooming_settings)


@attrs.frozen
class AngularitySpec(ParameterSpec):
    alpha: float
    kappa: float = attrs.field(default=1.0)

    def __str__(self) -> str:
        return f"Generalized Angularities(alpha={self.alpha}, kappa={self.kappa})"

    def encode(self) -> str:
        return f"alpha_{self.alpha}_kappa_{self.kappa}"

    @classmethod
    def decode(cls, value: str) -> AngularitySpec:
        # `value` is of the form: "alpha_{self.alpha}_kappa_{self.kappa}"
        # indices:                 0      1           2      3
        split = value.split("_")
        return cls(
            alpha=float(split[1]),
            kappa=float(split[3]),
        )


@attrs.frozen
class JetChargeSpec(ParameterSpec):
    kappa: float

    def __str__(self) -> str:
        return f"Jet charged (kappa={self.kappa})"

    def encode(self) -> str:
        return f"kappa_{self.kappa}"

    @classmethod
    def decode(cls, value: str) -> JetChargeSpec:
        # `value` is of the form: "kappa_{self.kappa}"
        # indices:                 0      1
        split = value.split("_")
        return cls(
            kappa=float(split[1]),
        )


@attrs.frozen
class SubjetRSpec(ParameterSpec):
    r: float

    def __str__(self) -> str:
        return f"Subjet Z_r(r={self.r})"

    def encode(self) -> str:
        return f"r_{self.r}"

    @classmethod
    def decode(cls, value: str) -> SubjetRSpec:
        # `value` is of the form: "r_{self.r}"
        # indices:                 0  1
        split = value.split("_")
        return cls(
            r=float(split[1]),
        )


@attrs.frozen
class EnergyCorrelatorWeightSpec(ParameterSpec):
    n: float

    def __str__(self) -> str:
        return f"Energy correlator weight(n={self.n})"

    def encode(self) -> str:
        return f"n_{self.n}"

    @classmethod
    def decode(cls, value: str) -> EnergyCorrelatorWeightSpec:
        # `value` is of the form: "n_{self.n}"
        # indices:                 0  1
        split = value.split("_")
        return cls(
            n=float(split[1]),
        )


def _decode_smearing_spec(value: str) -> PtSpec | EtSpec:
    """Helper to decode a smearing spec.

    Note:
        The types are indistinguishable just based on types along, so we need to encode the variable too.

    Args:
        value: Encoded string, exceptionally including the variable name
    """
    variable_name, *_ = value.split("_")
    spec_type = SpecDecoderRegistry.get_type(variable_name)
    # NOTE: +1 accounts for the trailing "_" separating the variable name from the values.
    return spec_type.decode(value[len(variable_name) + 1 :])


@attrs.frozen
class SmearingSpec(ParameterSpec):
    particle_level: PtSpec | EtSpec
    detector_level: PtSpec | EtSpec

    def __str__(self) -> str:
        return f"Smearing, part: {self.particle_level!s}, det: {self.detector_level!s}"

    def encode(self) -> str:
        # NOTE: Need to determine the type of each (PtSpec vs EtSpec) and encode that
        #       explicitly, since the encoding for a spec doesn't provide the name by default.
        return f"part_{SpecDecoderRegistry.get_name(self.particle_level)}_{self.particle_level.encode()}_det_{SpecDecoderRegistry.get_name(self.detector_level)}_{self.detector_level.encode()}"

    @classmethod
    def decode(cls, value: str) -> EtSpec:
        # `value` is of the form: "part_{type}_{encoded_particle_level}_det_{type}_{encoded_detector_level}"
        # This is trickier to decode than for other specs, so we need to handle it more explicitly.
        # First, let's split at "_det", and then we can handle the two separately.
        index_det = value.find("_det")
        # Removes "part_" to "_det"
        part_level = value[len("part_") : index_det]
        part_level_spec = _decode_smearing_spec(part_level)
        det_level = value[index_det + len("_det_") :]
        det_level_spec = _decode_smearing_spec(det_level)

        return cls(
            particle_level=part_level_spec,
            detector_level=det_level_spec,
        )


@attrs.frozen
class IsolationSpec(ParameterSpec):
    """Photon isolation specification.

    Attributes:
        type: Type of isolation, e.g. "neutral"
        R: Cone size of isolation (which we store in jet R for convenience, even if a jet finding isn't always used)
        Et_max_pp: Maximum Et allowed inside of the isolation cone in pp. If a single value is provided, it's assumed to
            be the max value.
        Et_max_AA: Maximum Et allowed inside of the isolation cone in AA. If a single value is provided, it's assumed to
            be the max value.
    """

    type: str
    R: JetRSpec = attrs.field(converter=lambda x: JetRSpec(x) if isinstance(x, float) else x)
    Et_max_pp: EtSpec = attrs.field(converter=lambda x: EtSpec(0, x) if isinstance(x, float) else x)
    Et_max_AA: EtSpec = attrs.field(converter=lambda x: EtSpec(0, x) if isinstance(x, float) else x)

    def __str__(self) -> str:
        return f"Photon Isolation (type={self.type}, R={self.R!s}, Et_max={self.Et_max!s})"

    def encode(self) -> str:
        return f"type_{self.type}_R_{self.R.encode()}_Et_max_pp_{self.Et_max_pp.encode()}_Et_max_AA_{self.Et_max_AA.encode()}"

    @classmethod
    def decode(cls, value: str) -> IsolationSpec:
        # `value` is of the form: "type_{self.type}_R_{self.R.encode()}_Et_max_pp_{self.Et_max_pp.encode()}_Et_max_AA_{self.Et_max_AA.encode()}"
        # indices:                 0     1          2  3                4  5   6   7                        8  9   10  11
        split = value.split("_")
        return cls(
            type=str(split[1]),
            R=JetRSpec.decode(value[value.find("_R_") + len("_R_") : value.find("_Et_max_pp")]),
            Et_max_pp=EtSpec.decode(value[value.find("_Et_max_pp") + len("_Et_max_pp_") : value.find("_Et_max_AA")]),
            Et_max_AA=EtSpec.decode(value[value.find("_Et_max_AA") + len("_Et_max_AA_") :]),
        )


T = TypeVar("T", bound=ParameterSpec)


@attrs.define
class ParameterSpecs(Generic[T]):
    """A set of parameter specifications.

    Attributes:
        values: Values of parameter spec
        name: Name of the parameter specs. By convention, this name should match the name of the parameter spec.
            It's also nice if it matches the name in the config, but that doesn't always make sense.
        spec_type: The ParameterSpec itself.
        label: An additional label to append to the
    """

    values: list[T]
    # NOTE: The name of the Variable must match the name of the ParameterSpec
    name: ClassVar[str] = "ParameterSpecs"
    spec_type: ClassVar[type[ParameterSpec]] = ParameterSpec
    label: str = attrs.field(default="")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Automatically extract and store the spec type
        if hasattr(cls, "__orig_bases__"):
            for base in cls.__orig_bases__:
                if get_origin(base) is ParameterSpecs:
                    args = get_args(base)
                    if args:
                        cls.spec_type = args[0]
                        break
        # Cross check
        if cls.spec_type is ParameterSpec:
            msg = f"Unable to determine ParameterSpec. Did you set it? Name: {cls.__name__}, {cls}"
            raise ValueError(msg)

        # We want to register the decoder for both the ParameterSpec and ParameterSpecs
        # Since we'll instantiate each of the ParameterSpecs if they're available, this
        # is a convenient way to register both.
        SpecDecoderRegistry.register(cls.name, cls.spec_type)
        SpecsDecoderRegistry.register(cls.name, cls)

    @property
    def encode_name(self) -> str:
        if self.label:
            return f"{self.label}_{self.name}"
        return self.name

    def encode(self) -> str:
        encoded = self.encode_name
        for v in self.values:
            v_encoded = v.encode() if isinstance(v, ParameterSpec) else str(v)  # type: ignore[redundant-expr]
            encoded += f"__{v_encoded}"
        return encoded

    @classmethod
    def decode(cls, value: str) -> ParameterSpecs[T]:
        # NOTE: The possible_label may be empty, but that's okay - it's the same as the default value
        possible_label, cleaned_value = cls._validate_and_extract_for_decode(value)

        values: list[T] = []
        for s in cleaned_value.split("__"):
            # Skip the case where we have the leading "__", which will resolve as an empty string.
            if s == "":
                continue

            # Decode into a Spec
            spec = SpecDecoderRegistry.decode(cls.name, s)
            # We know this is the correct type, so we're just helping out mypy
            values.append(cast(T, spec))

        return cls(
            values=values,
            label=possible_label,
        )

    @classmethod
    def _validate_and_extract_for_decode(cls, value: str) -> None:
        """Validate that the variable can decode this string, and extract possible relevant values.

        Args:
            value: Value to decode.
        Returns:
            If valid, returns the value to decode with the Variable tag removed
            for further processing.
        Raises:
            ValueError if the value cannot be decoded by the variable.
        """
        if cls.name not in value:
            msg = f"Asked to decode with {cls.name}, but missing '{cls.name}' label. Provided: {value}"
            raise ValueError(msg)
        # Expected format is:
        # - Without label: name__{value1}__{value2}__...
        # - With label: label_name__{value1}__{value2}__...
        # We can extract these based on the position of the name of the variable
        index_start_of_name = value.find(cls.name)
        index_end_of_name = index_start_of_name + len(cls.name)

        # Extract the values
        # It's up to the user to check whether the label is empty or not.
        # NOTE: the -1 offset is to account for the "_" between the label and the name.
        #       However, if there is no label, then we need to pass 0 (where the index_start_of_name points),
        #       or we'll incorrectly index far into the string.
        possible_label = value[: index_start_of_name - 1 if index_start_of_name > 0 else index_start_of_name]
        # We don't actually need the name itself, so just pass on the values
        cleaned_values = value[index_end_of_name:]
        return possible_label, cleaned_values

    def from_config(cls, config: dict[str, Any], label: str = "") -> ParameterSpecs[T]:
        """Initialize the parameter spec from the provided configuration.

        Args:
            config: Configuration to use to initialize the parameter specs
            label: Additional label for the spec, such as being a "trigger". This will be encoded
                with the parameter. Default: Empty string, which won't add any label.
        """
        raise NotImplementedError


# Decoder registry for parameter specs
SpecsDecoderRegistry = DecoderRegistry()


@attrs.define
class CentralitySpecs(ParameterSpecs[CentralitySpec]):
    name: ClassVar[str] = "centrality"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> CentralitySpecs:
        return cls(
            values=[CentralitySpec(*v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class PtSpecs(ParameterSpecs[PtSpec]):
    name: ClassVar[str] = "pt"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> PtSpecs:
        return cls(
            values=[PtSpec(*v) for v in itertools.pairwise(config[cls.name])],
            label=label,
        )


@attrs.define
class EtSpecs(ParameterSpecs[EtSpec]):
    name: ClassVar[str] = "Et"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> EtSpecs:
        return cls(
            values=[EtSpec(*v) for v in itertools.pairwise(config[cls.name])],
            label=label,
        )


@attrs.define
class EtaSpecs(ParameterSpecs[EtaSpec]):
    name: ClassVar[str] = "eta"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> EtaSpecs:
        value = config[cls.name]
        # Handle one or multiple arguments
        # In the case of multiple arguments, we generally put the min as the first value, so we want to assign that explicitly.
        # NOTE: We can't just do a `reversed()` since `min` is a keyword-only argument
        v = EtaSpec(value) if isinstance(value, float) else EtaSpec(**dict(zip(["min", "max"], value, strict=True)))
        return cls(
            values=[v],
            label=label,
        )


@attrs.define
class EtaRSpecs(ParameterSpecs[EtaSpec]):
    name: ClassVar[str] = "eta_R"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> EtaRSpecs:
        return cls(
            values=[EtaSpec(config[cls.name])],
            label=label,
        )


@attrs.define
class RapiditySpecs(ParameterSpecs[RapiditySpec]):
    name: ClassVar[str] = "rapidity"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> RapiditySpecs:
        return cls(
            values=[RapiditySpec(config[cls.name])],
            label=label,
        )


@attrs.define
class MassSpecs(ParameterSpecs[MassSpec]):
    name: ClassVar[str] = "mass"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> MassSpec:
        return cls(
            values=[MassSpec(*config[cls.name])],
            label=label,
        )


@attrs.define
class JetRSpecs(ParameterSpecs[JetRSpec]):
    name: ClassVar[str] = "R"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> JetRSpecs:
        return cls(
            values=[JetRSpec(v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class GroomingSettingsSpecs(ParameterSpecs[GroomingSettingsSpec]):
    name: ClassVar[str] = "grooming_settings"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> GroomingSettingsSpecs:
        return cls(
            values=[_convert_to_grooming_settings_spec(v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class JetAxisDifferenceSpecs(ParameterSpecs[JetAxisDifferenceSpec]):
    name: ClassVar[str] = "axis"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> JetAxisDifferenceSpecs:
        return cls(
            values=[JetAxisDifferenceSpec(**v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class AngularitySpecs(ParameterSpecs[AngularitySpec]):
    name: ClassVar[str] = "angularity"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> AngularitySpecs:
        return cls(
            values=[AngularitySpec(**v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class JetChargeSpecs(ParameterSpecs[JetChargeSpec]):
    name: ClassVar[str] = "charge"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> JetChargeSpecs:
        return cls(
            values=[JetChargeSpec(v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class SubjetRSpecs(ParameterSpecs[SubjetRSpec]):
    name: ClassVar[str] = "subjet_R"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> SubjetRSpecs:
        return cls(
            values=[SubjetRSpec(v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class EnergyCorrelatorWeightSpecs(ParameterSpecs[EnergyCorrelatorWeightSpec]):
    name: ClassVar[str] = "weight"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> EnergyCorrelatorWeightSpecs:
        return cls(
            values=[EnergyCorrelatorWeightSpec(v) for v in config[cls.name]],
            label=label,
        )


@attrs.define
class SmearingSpecs(ParameterSpecs[SmearingSpec]):
    """Detector-level smearing specifications

    NOTE:
        This case is a bit exceptional since it would need to align with the measured trigger range.
    """

    name: ClassVar[str] = "smearing"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> AngularitySpecs:
        values = []

        smearing_values = config[cls.name]
        for smearing_values in config[cls.name]:
            kwargs = {
                "particle_level": None,
                "detector_level": None,
            }
            for k in smearing_values:
                *_, variable_name = k.split("level_")
                spec_type = SpecDecoderRegistry.get_type(variable_name)
                for name in ["particle_level", "detector_level"]:
                    if name in k:
                        kwargs[name] = spec_type(*smearing_values)
            values.append(SmearingSpec(**kwargs))

        return cls(
            values=values,
            label=label,
        )


@attrs.define
class IsolationSpecs(ParameterSpecs[IsolationSpec]):
    name: ClassVar[str] = "isolation"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> IsolationSpecs:
        return cls(
            values=[IsolationSpec(**config[cls.name])],
            label=label,
        )


class ExtractParameters(Protocol):
    def __call__(self, config: dict[str, Any], label: str) -> AllParameters: ...


def extract_hadron_parameters(config: dict[str, Any], label: str) -> AllParameters:
    """Extract hadron parameters from the provided configuration.

    Args:
        config: (Possibly) hadron parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # There's no clear hadron proxy, so we'll assume that the user was responsible
    # and checked whether hadron fields are relevant.
    parameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        PtSpecs,
        EtaSpecs,
        RapiditySpecs,
    ]
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(specs_type.from_config(config, label=label))

    return parameters


def extract_hadron_correlations_parameters(config: dict[str, Any], label: str) -> AllParameters:
    """Extract hadron parameters from the provided configuration.

    Args:
        config: (Possibly) hadron parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # There's no clear hadron proxy, so we'll assume that the user was responsible
    # and checked whether hadron fields are relevant.
    # NOTE: This should be **just** customization that's distinct from hadron correlations.
    parameters: AllParameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        PtSpecs,
    ]
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(specs_type.from_config(config, label=label))

    return parameters


def extract_jet_parameters(config: dict[str, Any], label: str = "") -> AllParameters:
    """Extract jet parameters from the provided configuration.

    Args:
        config: (Possibly) jet parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # We'll use R as a proxy for a jet config being available since it's required.
    if "R" not in config:
        return []

    parameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        # Kinematics
        PtSpecs,
        EtaSpecs,
        EtaRSpecs,
        RapiditySpecs,
        # Jet finding parameters
        JetRSpecs,
        # Jet observable parameters
        GroomingSettingsSpecs,
        JetAxisDifferenceSpecs,
        AngularitySpecs,
        JetChargeSpecs,
        SubjetRSpecs,
        EnergyCorrelatorWeightSpecs,
    ]

    # Some types can be obviously related to the parameter category - e.g. soft drop is always related to a jet.
    # In that case, we can suppress the additional label.
    # This is mostly an aesthetic and conciseness choice, so as of 2026 April 16, we don't add any labels.
    # NOTE: If you modify this list, you should carefully follow through the consequences, and ensure that things still work.
    _suppress_label_for_these_types = []
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(
                specs_type.from_config(config, label=label if specs_type not in _suppress_label_for_these_types else "")
            )

    return parameters


def extract_pion_parameters(config: dict[str, Any], label: str) -> AllParameters:
    """Extract pion parameters from the provided configuration.

    Args:
        config: (Possibly) pion parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # There's no clear pion proxy, so we'll assume that the user was responsible
    # and checked whether the fields are relevant.
    parameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        # Kinematics
        PtSpecs,
        EtSpecs,
        EtaSpecs,
        RapiditySpecs,
        # Smearing
        SmearingSpecs,
    ]
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(specs_type.from_config(config, label=label))

    return parameters


def extract_gamma_parameters(config: dict[str, Any], label: str) -> AllParameters:
    """Extract gamma parameters from the provided configuration.

    Args:
        config: (Possibly) photon parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # There's no clear photon proxy, so we'll assume that the user was responsible
    # and checked whether the fields are relevant.
    parameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        # Kinematics
        PtSpecs,
        EtSpecs,
        EtaSpecs,
        RapiditySpecs,
        # Smearing
        SmearingSpecs,
        # Isolation
        IsolationSpecs,
    ]
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(specs_type.from_config(config, label=label))

    return parameters


def extract_z_parameters(config: dict[str, Any], label: str) -> AllParameters:
    """Extract Z boson parameters from the provided configuration.

    Args:
        config: (Possibly) Z boson parameters configuration.
        label: Additional label to add to the parameters, such as it being a trigger.

    Returns:
        Parameters that were found in the configuration.
    """
    # There's no clear Z-boson proxy, so we'll assume that the user was responsible
    # and checked whether the fields are relevant.
    parameters = []
    _relevant_spec_types: list[type[ParameterSpecs[Any]]] = [
        # Kinematics
        PtSpecs,
        EtSpecs,
        EtaSpecs,
        RapiditySpecs,
        MassSpecs,
    ]
    for specs_type in _relevant_spec_types:
        if specs_type.name in config:
            parameters.append(specs_type.from_config(config, label=label))

    # For additional parameters on electron and muon selections, we need to take another
    # step into the configuration
    additional_parameters = ["electron", "muon"]
    for k in additional_parameters:
        temp_label = f"{label}_{k}" if label else k
        parameters.extend(extract_hadron_parameters(config.get(k, {}), label=temp_label))

    return parameters


# All parameters that are relevant for an observable
AllParameters = list[ParameterSpecs[Any]]
# A selection of parameters that correspond to one configuration of one observable
Parameters = dict[str, str]
# The indices of a selection of parameters that correspond to one configuration of one observable (i.e. a ParameterSpec)
Indices = dict[str, int]

# Translate the encoded name into something more readable. It's not perfect, but good enough for most of our purposes.
_name_translation_map = {
    "pt_ch": "charged hadron RAA",
    "pt_pi0": "charged pion RAA",
    "IAA_pt": "IAA",
    "dphi": "acoplanarity (dphi)",
    "Dpt": "Fragmentation (Pt)",
    "Dz": "Fragmentation (z)",
    "ptd": "Dispersion (PtD)",
    "axis": "Jet-axis difference",
    "charge": "Jet charge",
    "ktg": "Groomed kt",
    "mg": "Groomed jet mass (Mg)",
    "rg ": "Groomed jet radius (Rg)",
    "tg": "Groomed jet radius (Rg)",
    "zg": "Groomed momentum fraction (zg)",
    "zr": "Subjet z",
    "angularity": "Generalized angularity",
    "mass": "Mass",
    "chjet": "charged-particle jet",
    "dihadron": "di-hadron correlations",
    "nsubjettiness": "N-subjettiness",
    "four": "four-particle cumulant",
    "sp": "scalar product",
}


def pretty_print_name(name: str) -> str:
    """Translates encoded name into something more readable.

    Args:
        name_substr: Name split into it's substrings for parsing. We don't do this
            automatically because sometimes you need to e.g. remove the experiment name.

    Returns:
        Readable name
    """
    working_str = name
    for k, v in _name_translation_map.items():
        if k in name:
            working_str = working_str.replace(k, v)
    return working_str.replace("_", " ")


_T_Spec = TypeVar("_T_Spec", bound=ParameterSpec)
_T_Specs = TypeVar("_T_Specs", bound=ParameterSpecs)


class DidNotFindDesiredParameterSpec(ValueError):
    """Indicates that we could not find the requested parameter spec."""


def find_parameter_by_spec_type(
    parameters: AllParameters,
    desired_type: type[_T_Spec] | type[_T_Specs],
) -> list[_T_Specs]:
    """Find the ParameterSpecs in a list of all observable parameters.

    It's preferred to iterate over values, but sometimes it's useful to be able to pick them out individually.

    Note:
        If you provide a ParameterSpec, we will still provide the full ParameterSpecs with all values.
        It will then be up to you to filter what you're interested in.

    Args:
        parameters: List of all observable parameters.
        desired_type: Type of the desired ParameterSpec of ParameterSpecs

    Returns:
        A list of the desired ParameterSpecs, or raises a ValueError if the type cannot be found
    """
    specs = []
    for p in parameters:
        if issubclass(desired_type, ParameterSpec):
            # Parameter Spec
            for v in p.values:
                if isinstance(v, desired_type):
                    specs.append(p)
                    # Since we return the full ParameterSpecs, we only want one copy
                    break
        elif isinstance(p, desired_type):
            # ParameterSpecs
            specs.append(p)

    if not specs:
        msg = f"Could not find desired type: {desired_type} in {parameters=}"
        raise DidNotFindDesiredParameterSpec(msg)

    return specs


@attrs.define
class Observable:
    sqrt_s: int = attrs.field()
    observable_class: str = attrs.field()
    name: str = attrs.field()
    config: dict[str, Any] = attrs.field()

    @property
    def identifier(self) -> tuple[float, str, str]:
        """Identify the observable."""
        return self.sqrt_s, self.observable_class, self.name

    @property
    def observable_str(self) -> str:
        """Observable identifier as a string."""
        return "_".join(map(str, self.identifier))

    @property
    def observable_str_as_path(self) -> Path:
        """Observable identifier as a path"""
        return Path(str(self.sqrt_s)) / self.observable_class / self.name

    @property
    def internal_name_without_experiment(self) -> str:
        """Internal observable name, excluding the experiment name."""
        *name, _ = self.name.split("_")
        return "_".join(name)

    @property
    def experiment(self) -> str:
        """Experiment which made this measurement."""
        return self.name.split("_")[-1].upper()

    @property
    def display_name(self) -> str:
        """Pretty print of the observable name.

        It's fairly ad-hoc, but at least gives us something to work with.
        """
        # -1 removes the experiment name
        return pretty_print_name(self.internal_name_without_experiment)

    def encode_name_for_storing_in_file(self, *, tag: str = "", **parameters_to_encode: ParameterSpec) -> str:
        # The baseline that we want to encode the essential parameters.
        # From there, we have two possible modifications:
        # 1. Additional parameters that we want to include (if they're available).
        additional_parameters_to_encode = [JetRSpec]
        # 2. Parameters that we want to exclude (namely, those which we don't want to specify at the observable level)
        #    For example, we skip the centrality since we want to deal with that binning later.
        parameters_to_exclude = [CentralitySpec]

        # We need the encoded parameter names to have the right names for comparison
        encoded_parameters = {p.encode_name: p for p in self.parameters()}
        encoded_essential_parameters = {p.encode_name: p for p in self.essential_parameters()}

        # Loop over the available parameters to order the additional parameters w.r.t the essential parameters
        # The order matters for the encoding
        to_encode = {}
        for encoded_name, param_specs in encoded_parameters.items():
            # Only include them if they're in the essential parameters, or we've specifically requested for them
            if not (encoded_name in encoded_essential_parameters or param_specs.spec_type in additional_parameters_to_encode):
                continue
            # And also skip those that we've excluded
            if param_specs.spec_type in parameters_to_exclude:
                continue

            try:
                to_encode[encoded_name] = parameters_to_encode.pop(encoded_name)
            except KeyError:
                msg = f"Expected to find {encoded_name}, but was not provided in {parameters_to_encode}. Please add it"
                raise KeyError(msg)

        if parameters_to_encode:
            logger.warning(f"Provided {parameters_to_encode=}, but they're not needed to encode the name")

        # Aiming for something like:
        # f"inclusive_chjet_ktg_alice_R{jetR}_zcut{zcut}_beta{beta}{jet_collection_label}"
        # TODO(RJE): So I need to encode the jet_R, the grooming setting, and the jet_collection_label
        # Or for tg:
        # inclusive_chjet_tg_alice_R0.2_zcut0.2_beta0
        base_name = f"{self.observable_class}_{self.name}"
        for k, v in to_encode.items():
            base_name += f"_{k}_{v.encode()}"
        # And then include if specified.
        if tag:
            base_name += f"_{tag}"

        # inclusive_chjet_ktg_alice_centrality_0_10_jet_R_0.2_jet_grooming_settings_DyG_a_1.0

        return base_name

    def parameters(self) -> AllParameters:
        """The parameter specifications that are relevant to the observable.

        NOTE:
            It's usually most convenient to access these values by iterating through a
            generator/list of combinations. However, it's also useful to be able to get
            the whole set together, so we provide this to the user too.

        Returns:
            Parameters.
        """
        ######################
        # Base parameters
        ######################
        # These are parameters that are not directly associated with a physics object
        # (e.g. not a hadron, jet, ...), such as centrality.
        _parameters: AllParameters = [CentralitySpecs.from_config(config=self.config)]

        ####################
        # Trigger parameters
        ####################
        # For the trigger / associated case, parameters are labeled by the quantities
        # i.e. hadron_pt, jet_pt, z_pt, ...
        if "trigger" in self.observable_class:
            trigger_to_parameter_specs: dict[str, ExtractParameters] = {
                "hadron": extract_hadron_parameters,
                "dijet": extract_jet_parameters,
                "pion": extract_pion_parameters,
                "gamma": extract_gamma_parameters,
                "z": extract_z_parameters,
            }
            for trigger_name, func in trigger_to_parameter_specs.items():
                if f"{trigger_name}_trigger" in self.observable_class:
                    # The config will always be named "trigger" regardless of the type, so we can
                    # always ask for the same name here.
                    res = func(config=self.config.get("trigger", {}), label=f"{trigger_name}_trigger")
                    _parameters.extend(res)

        ###############################################
        # Inclusive and/or recoil/associated parameters
        ###############################################
        # We cover two cases here:
        # 1. Inclusive hadrons and jets properties
        # 2. Recoil/associated properties from a trigger
        # NOTE: The observable_class are of the form "X_trigger_Y", where X is the trigger and Y is the recoil/associated.
        #       Since we want to target the recoil here, we look for "_trigger_Y" for Y

        # Hadron parameters
        if self.observable_class == "hadron" or "_trigger_hadron" in self.observable_class:
            _parameters.extend(extract_hadron_parameters(config=self.config.get("hadron", {}), label="hadron"))

        # Inclusive jet parameters
        if (
            self.observable_class in ["inclusive_chjet", "inclusive_jet"]
            or "_trigger_chjet" in self.observable_class
            or "_trigger_jet" in self.observable_class
        ):
            _parameters.extend(extract_jet_parameters(config=self.config.get("jet", {}), label="jet"))

        ################################
        # Hadron correlations parameters
        ################################
        # This isn't necessarily a traditional trigger particle here, so we just handle it as an inclusive case.
        # NOTE: This method probably needs to be developed further
        if self.observable_class == "hadron_correlation":
            _parameters.extend(extract_hadron_correlations_parameters(config=self.config, label=""))

        return _parameters

    def essential_parameters(self) -> AllParameters:
        """Just the parameters that are needed to describe variations in an observable.

        In practice, this means parameters that actually vary, rather than those where there is only one variation.

        Args:
            None. It will determine the observable parameters
        Returns:
            Just the essential parameters needed to describe observable variations.
        """
        return [p for p in self.parameters() if len(p.values) > 1]

    def generate_parameter_combinations(self, parameters: AllParameters) -> Iterator[tuple[Parameters, Indices]]:
        """Generate combinations of parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            List of parameter specs, list of bins associated with the parameters (e.g. pt). These
            can be zipped together to provide the appropriate indices.
        """
        # Determining the appropriate label is a bit tricky:
        # 1. We want to be able to identify the parameter by the obvious name - e.g. `pt`.
        #    This would argue for the parameter spec name.
        # 2. However, we need unique names for the keys, since e.g. both the trigger and the recoil could have a pt field.
        #    This would argue for using the encoded name.
        # As of 2026 April, we return #2 since we need the unique key. But if this became a problem,
        # we could always add another mapping between the key and the "obvious" parameter name.

        # Need the names for labeling later
        # As noted above, we're return the encoded name since it's unique.
        parameter_specs_names = [p.encode_name for p in parameters]
        # parameter_specs_names = [p.name for p in parameters]
        # And then determine all of the values, as well as the indices
        values_of_parameters = [p.values for p in parameters]
        indices = [list(range(len(p))) for p in values_of_parameters]
        # Get all combinations
        combinations = itertools.product(*values_of_parameters)
        indices_combinations = itertools.product(*indices)

        # And then return them labeled by the parameter name
        # We want: ({"pt": [], ...], {"pt": }})
        # NOTE: Since we use the encoded parameters, the key will likely be more complicated - e.g. `jet_pt`
        yield from zip(
            (dict(zip(parameter_specs_names, values, strict=True)) for values in combinations),
            (dict(zip(parameter_specs_names, values, strict=True)) for values in indices_combinations),
            strict=True,
        )

    def parameter_combinations_data_frame(self) -> pd.DataFrame:
        """Define a dataframe with all of the parameters.

        NOTE:
            This is _NOT_ a high performance data frame. However, it's quite convenient for representing
            our parameter combinations, so we live with it.

        Returns:
            Dataframe with one row per parameter combination.
        """
        # Delayed import to reduce dependence
        import pandas as pd  # noqa: PLC0415

        g = self.generate_parameter_combinations(self.parameters())

        # Retrieve the parameter combinations by label
        data = [v[0] for v in g]

        # And then convert into dataframe
        return pd.DataFrame(data)


def read_observables_from_config(config_path: Path, sqrt_s: int) -> dict[str, Observable]:
    with config_path.open() as f:
        config = yaml.safe_load(f)

    # We need to skip all of the parameters at the top of the config files
    observable_classes = []
    found_start_of_observables = False
    for k in config:
        if "hadron" in k or found_start_of_observables:
            observable_classes.append(k)
            found_start_of_observables = True

    # Now extract all of the observables
    observables = {}
    for observable_class in observable_classes:
        for observable_key in config[observable_class]:
            observable_info = config[observable_class][observable_key]
            observables[f"{sqrt_s}_{observable_class}_{observable_key}"] = Observable(
                sqrt_s=sqrt_s,
                observable_class=observable_class,
                name=observable_key,
                config=observable_info,
            )

    return observables


def read_observables_from_all_config(
    jetscape_analysis_config_path: Path, sqrt_s_values_to_load: list[int] | None = None
) -> dict[str, Observable]:
    """Construct Observables from the configuration files at a given path.

    Note:
        This aggregation is built for STAT, so we assume that we're looking for STAT configs.

    Args:
        jetscape_analysis_config_path: Path to the configuration directory.
    Returns:
        A mapping of observable_str -> Observable built from all configuration files.
    """
    # Validation
    jetscape_analysis_config_path = Path(jetscape_analysis_config_path)

    # Parameters
    if sqrt_s_values_to_load is None:
        sqrt_s_values_to_load = [200, 2760, 5020]

    # Read configuration files
    observables = {}
    for sqrt_s in sqrt_s_values_to_load:
        observables.update(
            read_observables_from_config(
                config_path=jetscape_analysis_config_path / f"STAT_{sqrt_s}.yaml",
                sqrt_s=sqrt_s,
            )
        )

    # Return them sorted by convention
    return sort_observables(observables)


def sort_observables(observables: dict[str, Observable]) -> dict[str, Observable]:
    """Convenience function for providing sorted observables"""
    return dict(sorted(observables.items(), key=lambda o: (o[1].sqrt_s, o[1].observable_class, o[1].name)))


def main(jetscape_analysis_config_path: Path) -> None:
    """Just some testing and development code.

    Not for actual use...
    """
    # Just some testing code...
    observables = read_observables_from_all_config(jetscape_analysis_config_path=jetscape_analysis_config_path)

    # for obs in sorted(observables):
    # for sqrt_s in sorted(observables.keys()):
    for obs in observables.values():
        # Group by observable class
        # class_to_obs: dict[str, list[Observable]] = {}
        # for obs in observables[sqrt_s].values():
        #    class_to_obs.setdefault(obs.observable_class, []).append(obs)
        # for obs_class_name, obs_class in class_to_obs.items():
        # for obs in sorted(obs_class, key=lambda o: (o.sqrt_s, o.observable_class, o.name)):
        full_set_of_parameters = obs.parameters()

        # if "trigger" in obs.observable_class:
        if "ktg" in obs.name:
            logger.info(f"{full_set_of_parameters=}")
            # fmt:off
            import IPython; IPython.embed()  # noqa: PLC0415,I001,E702
            import sys; sys.exit(1)  # noqa: PLC0415,I001,E702
            # fmt:on


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
        help="Path to the jetscape-analysis config directory.",
        required=True,
    )
    args = parser.parse_args()

    main(jetscape_analysis_config_path=args.jetscape_analysis_config)
