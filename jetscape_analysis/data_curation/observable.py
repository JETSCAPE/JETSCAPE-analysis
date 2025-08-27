"""Represent the information about a single Observable.

An Observable is measured for a set parameters, such as jet_R, pt, etc.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
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
class EtaSpec(ParameterSpec):
    value: float

    def __str__(self) -> str:
        return f"|eta|<={self.value}"

    def encode(self) -> str:
        return f"{self.value!s}"

    @classmethod
    def decode(cls, value: str) -> EtaSpec:
        # `value` is of the form: "{self.value}"
        return cls(
            value=float(value),
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

    def encode_name(self) -> str:
        if self.label:
            return f"{self.label}_{self.name}"
        return self.name

    def encode(self) -> str:
        encoded = self.name
        for v in self.values:
            v_encoded = v.encode() if isinstance(v, ParameterSpec) else str(v)  # type: ignore[redundant-expr]
            encoded += f"__{v_encoded}"
        return encoded

    @classmethod
    def decode(cls, value: str) -> ParameterSpecs[T]:
        cleaned_value = cls._decode_validity_check(value)
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
class EtaSpecs(ParameterSpecs[EtaSpec]):
    name: ClassVar[str] = "eta"

    @classmethod
    def from_config(cls, config: dict[str, Any], label: str = "") -> EtaSpecs:
        return cls(
            values=[EtaSpec(config["eta_cut"])],
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


class ExtractParameters(Protocol):
    def __call__(self, config: dict[str, Any], label: str) -> AllParameters: ...


def extract_hadron_parameters(config: dict[str, Any], label: str) -> AllParameters:
    # There's no clear hadron proxy, so we'll assume that the user was responsible
    # and checked whether hadron fields are relevant.
    parameters = []
    _field_to_specs_type: dict[str, type[ParameterSpecs[Any]]] = {
        "pt": PtSpecs,
        "eta_cut": EtaSpecs,
    }
    for field_name, specs_type in _field_to_specs_type.items():
        if field_name in config:
            parameters.append(specs_type.from_config(config, label=label))

    return parameters


def extract_hadron_correlations_parameters(config: dict[str, Any]) -> AllParameters:
    # There's no clear hadron proxy, so we'll assume that the user was responsible
    # and checked whether hadron fields are relevant.
    parameters: AllParameters = []
    _field_to_specs_type: dict[str, type[ParameterSpecs[Any]]] = {
        "pt": PtSpecs,
    }
    for field_name, specs_type in _field_to_specs_type.items():
        if field_name in config:
            parameters.append(specs_type.from_config(config))

    return parameters


def extract_jet_parameters(config: dict[str, Any], label: str = "") -> AllParameters:
    # We'll use jet_R as a proxy for a jet config being available since it's required
    if "jet_R" not in config:
        return []

    parameters = []
    _field_to_specs_type: dict[str, type[ParameterSpecs[Any]]] = {
        "jet_R": JetRSpecs,
        "pt": PtSpecs,
        "soft_drop": SoftDropSpecs,
        "dynamical_grooming": DynamicalGroomingSpecs,
        "axis": JetAxisDifferenceSpecs,
        "kappa": AngularitySpecs,
        "r": SubjetZSpecs,
    }
    for field_name, specs_type in _field_to_specs_type.items():
        if field_name in config:
            parameters.append(specs_type.from_config(config, label=label))

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

    def parameters(self) -> AllParameters:
        """The parameter specifications that are relevant to the observable.

        NOTE:
            It's usually most convenient to access these values by iterating through a
            generator/list of combinations. However, it's also useful to be able to get
            the whole set together, so we provide this to the user too.

        Returns:
            Parameters.
        """
        # Base parameters
        _parameters: AllParameters = [CentralitySpecs.from_config(config=self.config)]

        ######################
        # Inclusive parameters
        ######################
        if self.observable_class.startswith("inclusive") or self.observable_class == "hadron":
            # Jet parameters
            if "jet" in self.observable_class:
                _parameters.extend(extract_jet_parameters(config=self.config, label=""))

            # Hadron parameters
            if "hadron" in self.observable_class:
                _parameters.extend(extract_hadron_parameters(config=self.config, label=""))

        # Hadron correlations
        if self.observable_class == "hadron_correlation":
            _parameters.extend(extract_hadron_correlations_parameters(config=self.config))

        ####################
        # Trigger parameters
        ####################
        # For the trigger / associated case, parameters are labeled by the quantities
        # i.e. hadron_pt, jet_pt, z_pt, ...
        # TODO(RJE): Need to handle the inclusive case prefix too(?)
        if "trigger" in self.observable_class:
            trigger_to_parameter_specs: dict[str, ExtractParameters] = {
                "hadron": extract_hadron_parameters,
                "dijet": extract_jet_parameters,
                # TEMP: Use hadron just for testing so I can run the code
                # "pion": extract_pion_parameters,
                # "gamma": extract_gamma_parameters,
                # "z": extract_z_parameters,
                "pion": extract_hadron_parameters,
                "gamma": extract_hadron_parameters,
                "z": extract_hadron_parameters,
                # ENDTEMP
            }
            for trigger_name, func in trigger_to_parameter_specs.items():
                if f"{trigger_name}_trigger" in self.observable_class:
                    # The config will always be named "trigger" regardless of the type, so we can
                    # always ask for the same name here.
                    res = func(config=self.config.get("trigger", {}), label=f"{trigger_name}_trigger")
                    _parameters.extend(res)

            # And then
            # Recoil/associated properties
            # NOTE: The observable_class are of the form "X_trigger_Y", where X is the trigger and Y is the recoil/associated.
            #       Since we only want to target the recoil here, we look for "_trigger_Y" for Y
            # Jet properties
            # NOTE: Need to explicitly check for chjet and jet since we're including more of the observable class
            if "_trigger_chjet" in self.observable_class or "_trigger_jet" in self.observable_class:
                # Either ch_jet or jet
                jet_label = self.observable_class.split("_")[-1]
                _parameters.extend(extract_jet_parameters(config=self.config.get("jet", {}), label=jet_label))

            # Hadron properties
            if "_trigger_hadron" in self.observable_class:
                _parameters.extend(extract_hadron_parameters(config=self.config.get("hadron", {}), label="hadron"))

        return _parameters

    def generate_parameter_combinations(self, parameters: AllParameters) -> Iterator[tuple[Parameters, Indices]]:
        """Generate combinations of parameters that are relevant to the observable.

        Note:
            The bin indices are not meant to be comprehensive - just those that are
            relevant for the observable.

        Returns:
            List of parameter specs, list of bins associated with the parameters (e.g. pt). These
            can be zipped together to provide the appropriate indices.
        """
        # Need the names for labeling later
        parameter_specs_names = [p.encode_name() for p in parameters]
        # And then determine all of the values, as well as the indices
        values_of_parameters = [p.values for p in parameters]
        indices = [list(range(len(p))) for p in values_of_parameters]
        # Get all combinations
        combinations = itertools.product(*values_of_parameters)
        indices_combinations = itertools.product(*indices)

        # And then return them labeled by the parameter name
        # We want: ({"pt": [], ...], {"pt": }})
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


def main(jetscape_analysis_config_path: Path) -> None:
    import yaml

    # Parameters
    sqrt_s_values = [200, 2760, 5020]

    # Read configuration files
    configs = {}
    observable_classes = {}
    for sqrt_s in sqrt_s_values:
        with (jetscape_analysis_config_path / f"STAT_{sqrt_s}.yaml").open() as f:
            configs[sqrt_s] = yaml.safe_load(f)

        observable_classes[sqrt_s] = []
        found_start_of_observables = False
        for k in configs[sqrt_s]:
            if "hadron" in k or found_start_of_observables:
                observable_classes[sqrt_s].append(k)
                found_start_of_observables = True

    # Now extract all of the observables
    observables = {}
    for sqrt_s, config in configs.items():
        observables[sqrt_s] = {}
        for observable_class in observable_classes[sqrt_s]:
            for observable_key in config[observable_class]:
                observable_info = config[observable_class][observable_key]
                observables[sqrt_s][f"{observable_class}_{observable_key}"] = Observable(
                    sqrt_s=sqrt_s,
                    observable_class=observable_class,
                    name=observable_key,
                    config=observable_info,
                )

    # Just some testing code...
    for sqrt_s in sorted(observables.keys()):
        output_line_base = f"{sqrt_s}"
        # Group by observable class
        class_to_obs: dict[str, list[Observable]] = {}
        for obs in observables[sqrt_s].values():
            class_to_obs.setdefault(obs.observable_class, []).append(obs)
        for obs_class_name, obs_class in class_to_obs.items():
            for obs in sorted(obs_class, key=lambda o: o.name):
                base_values = [
                    output_line_base,
                    obs_class_name,
                    obs.internal_name_without_experiment,
                    obs.display_name,
                    obs.experiment,
                ]
                columns_to_print_separately = ["centrality"]
                full_set_of_parameters = obs.parameters()

                if "trigger" in obs_class_name:
                    logger.info(f"{full_set_of_parameters=}")
                    import IPython

                    IPython.embed()
                    import sys

                    sys.exit(1)


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
