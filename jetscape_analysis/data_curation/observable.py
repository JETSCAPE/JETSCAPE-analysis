"""Represent the information about a single Observable.

An Observable is measured for a set parameters, such as jet_R, pt, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, get_args, get_origin, runtime_checkable

import attrs
import numpy as np

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
        if name not in self._registry:
            msg = f"Unknown decoder type: {name}"
            raise ValueError(msg)
        return self._registry[name]

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
    z_cut: float
    beta: float

    def __str__(self) -> str:
        return f"Soft Drop(z_cut={self.z_cut}, beta={self.beta})"

    def encode(self) -> str:
        return f"z_cut_{self.z_cut}_beta_{self.beta}"

    @classmethod
    def decode(cls, value: str) -> SoftDropSpec:
        # `value` is of the form: "z_cut_{self.z_cut}_beta_{self.beta}"
        # indices:                 0 1    2           3     4
        split = value.split("_")
        return cls(
            z_cut=float(split[2]),
            beta=float(split[4]),
        )


@attrs.frozen
class DynamicalGroomingSpec(ParameterSpec):
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


@attrs.frozen
class JetAxisDifferenceSpec(ParameterSpec):
    type: str
    grooming_settings: SoftDropSpec | None = None

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
        # where type is of the form method1_method2, so the indices are offset by one
        # indices:                 0 (1)   2
        split = value.split("_")
        grooming_settings = None
        if len(split) > 2:
            # We have grooming settings, so we need to parse those
            # In practice, it's always using SoftDrop as of August 2025, so we'll live with a hard code here
            grooming_settings = SoftDropSpec.decode("_".join(split[2:]))

        return cls(type="_".join(split[:2]), grooming_settings=grooming_settings)


@attrs.frozen
class AngularitySpec(ParameterSpec):
    kappa: float

    def __str__(self) -> str:
        return f"Generalized Angularities(kappa={self.kappa})"

    def encode(self) -> str:
        return f"kappa_{self.kappa}"

    @classmethod
    def decode(cls, value: str) -> AngularitySpec:
        # `value` is of the form: "kappa_{self.kappa}"
        # indices:                 0      1
        split = value.split("_")
        return cls(
            kappa=float(split[1]),
        )


@attrs.frozen
class SubjetZSpec(ParameterSpec):
    r: float

    def __str__(self) -> str:
        return f"Subjet Z_r(r={self.r})"

    def encode(self) -> str:
        return f"r_{self.r}"

    @classmethod
    def decode(cls, value: str) -> SubjetZSpec:
        # `value` is of the form: "r_{self.r}"
        # indices:                 0  1
        split = value.split("_")
        return cls(
            r=float(split[1]),
        )


T = TypeVar("T", bound=ParameterSpec)


@attrs.define
class ParameterSpecs(Generic[T]):
    values: list[T]
    # NOTE: The name of the Variable must match the
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

    def encode(self) -> str:
        encoded = self.name
        for v in self.values:
            v_encoded = v.encode() if isinstance(v, ParameterSpec) else str(v)
            encoded += f"__{v_encoded}"
        return encoded

    @classmethod
    def decode(cls, value: str) -> ParameterSpecs[T]:
        cleaned_value = cls._decode_validity_check(value)
        # values: list[SoftDropSpec] = []
        values: list[T] = []
        for s in cleaned_value.split("__"):
            spec = SpecDecoderRegistry.decode(cls.name, s)
            # We know this is the correct type, so we're just helping out mypy
            values.append(cast(T, spec))

        return cls(values=values)

    @classmethod
    def _decode_validity_check(cls, value: str) -> str:
        """Check that the variable is valid to decode the value.

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
        return value[value.find(cls.name) + len(cls.name) :]

    def from_config(cls, config: dict[str, Any], label: str = "") -> ParameterSpecs[T]:
        raise NotImplementedError


# Decoder registry for parameter specs
SpecsDecoderRegistry = DecoderRegistry()


@attrs.define
class CentralitySpecs(ParameterSpecs[CentralitySpec]):
    name: ClassVar[str] = "centrality"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> CentralitySpecs:
        return cls(
            values=[CentralitySpec(*v) for v in config["centrality"]],
            label=label,
        )


@attrs.define
class PtSpecs(ParameterSpecs[PtSpec]):
    name: ClassVar[str] = "pt"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> PtSpecs:
        return cls(
            values=[PtSpec(*v) for v in itertools.pairwise(config["pt"])],
            label=label,
        )


@attrs.define
class JetRSpecs(ParameterSpecs[JetRSpec]):
    name: ClassVar[str] = "jet_R"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> JetRSpecs:
        return cls(
            values=[JetRSpec(v) for v in config["jet_R"]],
            label=label,
        )


@attrs.define
class SoftDropSpecs(ParameterSpecs[SoftDropSpec]):
    name: ClassVar[str] = "soft_drop"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> SoftDropSpecs:
        return cls(
            values=[SoftDropSpec(**v) for v in config["soft_drop"]],
            label=label,
        )


@attrs.define
class DynamicalGroomingSpecs(ParameterSpecs[DynamicalGroomingSpec]):
    name: ClassVar[str] = "dynamical_grooming"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> DynamicalGroomingSpecs:
        return cls(
            values=[DynamicalGroomingSpec(**v) for v in config["dynamical_grooming"]],
            label=label,
        )


@attrs.define
class JetAxisDifferenceSpecs(ParameterSpecs[JetAxisDifferenceSpec]):
    name: ClassVar[str] = "axis"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> JetAxisDifferenceSpecs:
        return cls(
            values=[JetAxisDifferenceSpec(**v) for v in config["axis"]],
            label=label,
        )


@attrs.define
class AngularitySpecs(ParameterSpecs[AngularitySpec]):
    name: ClassVar[str] = "kappa"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> AngularitySpecs:
        return cls(
            values=[AngularitySpec(v) for v in config["kappa"]],
            label=label,
        )


@attrs.define
class SubjetZSpecs(ParameterSpecs[SubjetZSpec]):
    name: ClassVar[str] = "subjet_z"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> SubjetZSpecs:
        return cls(
            values=[SubjetZSpec(v) for v in config["r"]],
            label=label,
        )


# All parameters that are relevant for an observable
AllParameters = dict[str, list[Any]]
# A selection of parameters that correspond to one configuration of one observable
Parameters = dict[str, Any]
# The indices of a selection of parameters that correspond to one configuration of one observable
Indices = dict[str, int]


class ParameterGroup(Protocol):
    """Group of parameters.

    This is an interface that each group of parameters must implement.
    """

    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]: ...

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]: ...


class BaseParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        base_parameters = {}
        # Centrality
        base_parameters["centrality"] = [tuple(v) for v in config["centrality"]]
        return base_parameters

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        # Centrality
        cent_low, cent_high = parameters["centrality"]
        output_parameters["centrality"] = f"{cent_low}-{cent_high}%"

        return output_parameters


class PtParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        values = []
        if "pt" in config:
            pt_values = config["pt"]
            values = [(pt_low, pt_high) for pt_low, pt_high in itertools.pairwise(pt_values)]
        # Removed as an option in August 2025
        # elif "pt_min" in config:
        #     values = [(config["pt_min"], -1)]

        if values:
            # Wrap it in a "pt" key to handle it similarly to the other parameters
            return {"pt": values}
        # Nothing to construct out of this
        return {}

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        if "pt" in parameters:
            pt_low, pt_high = parameters["pt"]
            if pt_high == -1:
                output_parameters["pt"] = f"pt > {pt_low}"
            else:
                output_parameters["pt"] = f"{pt_low} < pt < {pt_high}"
        # return _propagate_rest_of_parameters(output_parameters=output_parameters, parameters=parameters)
        return output_parameters


class JetParameters:
    def construct_parameters(observable: Observable, config: dict[str, Any]) -> dict[str, list[Any]]:
        values = {}

        # Handle standard cases first
        # Standardize parameter names
        parameter_names = {
            "jet_R": "jet_R",
            "axis": "axis",
            "kappa": "kappa",
            "r": "subjet_zr",
            "SoftDrop": "soft_drop",
            "dynamical_grooming_a": "dynamical_grooming",
        }
        for input_name, output_name in parameter_names.items():
            if input_name in config:
                values[output_name] = config[input_name]
        # Finally, handle special cases:
        # i.e. the pt
        values.update(PtParameters.construct_parameters(observable=observable, config=config))

        return values

    def format_parameters_for_printing(parameters: dict[str, list[Any]]) -> dict[str, list[str]]:
        # output_parameters = collections.defaultdict(list)
        output_parameters = {}
        # Jet R
        if "jet_R" in parameters:
            output_parameters["jet_R"] = f"R={parameters['jet_R']}"
        # pt
        if "pt" in parameters:
            output_parameters.update(PtParameters.format_parameters_for_printing(parameters=parameters))
        # Axis
        if "axis" in parameters:
            param = parameters["axis"]
            description = f"{param['type']}"
            if "grooming_settings" in param:
                description += (
                    f", SD z_cut={param['grooming_settings']['zcut']}, beta={param['grooming_settings']['beta']}"
                )
            output_parameters["axis"] = description
        # Kappa
        if "kappa" in parameters:
            output_parameters["kappa"] = f"ang. kappa={parameters['kappa']}"
        # Subjet z
        if "subjet_zr" in parameters:
            output_parameters["subjet_zr"] = f"Subjet r={parameters['subjet_zr']}"
        # Grooming
        # Soft Drop
        if "soft_drop" in parameters:
            # output_parameters["soft_drop"].extend(f"SD z_cut={param['zcut']}, beta={param['beta']}" for param in parameters["soft_drop"])
            param = parameters["soft_drop"]
            output_parameters["soft_drop"] = f"SD z_cut={param['zcut']}, beta={param['beta']}"
        # DyG
        if "dynamical_grooming" in parameters:
            # output_parameters["dynamical_grooming"].extend(f"DyG a={a}" for a in parameters["dynamical_grooming"])
            output_parameters["dynamical_grooming"] = f"DyG a={parameters['dynamical_grooming']}"

        return output_parameters


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


@attrs.define
class Observable:
    sqrt_s: float = attrs.field()
    observable_class: str = attrs.field()
    name: str = attrs.field()
    config: dict[str, Any] = attrs.field()

    @property
    def identifier(self) -> str:
        return f"{self.observable_class}_{self.name}"

    @property
    def internal_name_without_experiment(self) -> str:
        *name, _ = self.name.split("_")
        return "_".join(name)

    @property
    def experiment(self) -> str:
        return self.name.split("_")[-1].upper()

    @property
    def display_name(self) -> str:
        """Pretty print of the observable name.

        It's fairly ad-hoc, but at least gives us something to work with.
        """
        # -1 removes the experiment name
        return pretty_print_name("_".join(self.name.split("_")[:-1]))

    def inspire_hep_identifier(self) -> tuple[str, int]:
        """Extract InspireHEP identifier from the config if possible."""
        # Attempt to extract from the HEPdata filename.

        # Validation
        # We mostly don't care about the pp HEPdata - it's mostly about the AA
        if not ("hepdata" in self.config or "hepdata_AA" in self.config):
            msg = f"Cannot find HEPdata key for observable {self.identifier}"
            raise ValueError(msg)

        hepdata_key = "hepdata"
        if hepdata_key not in self.config:
            hepdata_key = "hepdata_AA"

        # Example: "HEPData-ins1127262-v2-root.root"
        hepdata = self.config[hepdata_key]
        _, hepdata_id, hepdata_version, *_ = hepdata.split("-")
        # Remove "ins"
        hepdata_id = hepdata_id.replace("ins", "")
        # Extract just the numerical version number
        hepdata_version = hepdata_version.replace("v", "")

        return hepdata_id, int(hepdata_version)

    def parameters(self) -> tuple[list[Parameters], list[Indices]]:
        """The parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            Parameters, bin indices associated with the parameters (e.g. pt).
        """
        _parameters = BaseParameters.construct_parameters(observable=self, config=self.config)
        # TODO(RJE): Handle trigger appropriately...
        if "trigger" in self.observable_class:
            ...

        if "jet_R" in self.config:
            _parameters.update(JetParameters.construct_parameters(observable=self, config=self.config))

        return _parameters

    def format_parameters_for_printing(self, parameters: dict[str, Any]) -> dict[str, str]:
        output_parameters = BaseParameters.format_parameters_for_printing(parameters)
        if "jet_R" in self.config:
            output_parameters.update(JetParameters.format_parameters_for_printing(parameters))
        # TODO(RJE): Handle trigger appropriately...

        missing_keys = set(parameters).difference(set(output_parameters))
        if missing_keys:
            logger.warning(f"missing formatting for {missing_keys}")

        # NOTE: Wrapped in dict to avoid leaking the defaultdict
        return dict(output_parameters)

    def generate_parameters(self, parameters: dict[str, list[Any]]) -> Iterator[tuple[str, dict[str, int]]]:
        """Generate combinations of parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            Description of parameters, bins associated with the parameters (e.g. pt).
        """
        # Add indices before each parameters:
        # e.g. "pt": [[1, 2], [2, 3]] -> [(0, [1,2]), (1, [2,3])]
        # parameters_with_indices = {
        #    p: [(i, v) for i, v in enumerate(values)] for p, values in parameters.items()
        # }
        # TODO(RJE): Grooming parameters are mutually exclusive, so we need to handle them one-by-one
        # grooming_parameters
        # We get the same combinations, but also with the indices.
        indices = {p: list(range(len(values))) for p, values in parameters.items()}
        # Get all combinations
        combinations = itertools.product(*parameters.values())
        indices_combinations = itertools.product(*indices.values())

        # And then return them labeled by the parameter name
        # We want: ({"pt": [], ...], {"pt": }})
        yield from zip(
            (dict(zip(parameters.keys(), values, strict=True)) for values in combinations),
            (dict(zip(parameters.keys(), values, strict=True)) for values in indices_combinations),
            strict=True,
        )

    def to_markdown(self, name_prefix: str | None = None) -> str:  # noqa: C901
        """Return a pretty, formatted markdown string for this observable."""
        display_name = self.display_name
        if name_prefix is not None:
            display_name = f"{name_prefix} {display_name}"
        lines = [f"- **Name:** {display_name}", f"  - **Experiment:** {self.experiment}"]
        try:
            hep_id, hep_version = self.inspire_hep_identifier()
            lines.append(f"  - **InspireHEP ID:** {hep_id} (v{hep_version})")
        except Exception:
            lines.append("  - **InspireHEP ID:** Not found")
        implementation_status = "Work-in-progress"
        if self.config["enabled"]:
            implementation_status = "Implemented"
        lines.append(f"  - **Implementation status**: {implementation_status}")

        # Special handling for particular keys:
        parameters = ["jet_R", "kappa", "axis", "SoftDrop"]
        if any(parameter in self.config for parameter in parameters):
            lines.append("  - **Parameters**:")
            if "jet_R" in self.config:
                lines.append(f"     - Jet R: {self.config['jet_R']}")
            if "kappa" in self.config:
                lines.append(f"     - Angularity kappa: {self.config['kappa']}")
            if "axis" in self.config:
                lines.append("     - Jet-axis difference:")
                for parameters in self.config["axis"]:
                    description = f"{parameters['type']}"
                    if "grooming_settings" in parameters:
                        description += f", z_cut={parameters['grooming_settings']['zcut']}, beta={parameters['grooming_settings']['beta']}"
                    lines.append(f"       - {description}")
            if "SoftDrop" in self.config:
                lines.append("     - SoftDrop:")
                for parameters in self.config["SoftDrop"]:
                    lines.append(f"       - z_cut={parameters['zcut']}, beta={parameters['beta']}")
            if "dynamical_grooming_a" in self.config:
                lines.append("     - Dynamical grooming:")
                for param in self.config["dynamical_grooming_a"]:
                    lines.append(f"       - a = {param}")

        return "\n".join(lines)

    def to_csv_like(self, separator: str = "\t") -> str:
        """Convert observable into a csv-like entries.

        If there are multiple parameters that would justify multiple entries,
        then multiple lines are provided.

        Args:
            separator: Separator used to differentiate fields. Default: `\t`.
        Returns:
            Lines formatted suitably for a csv-like.
        """
