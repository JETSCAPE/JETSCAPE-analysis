"""
Validation utilities for analysis YAML configuration files using Pydantic.

This module provides comprehensive validation for JETSCAPE analysis configuration files
using Pydantic models for type safety and validation.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from jetscape_analysis.base import helpers

logger = logging.getLogger(__name__)


# Translate names as needed
_name_translation_map = {}


def pretty_print_name(name: str) -> str:
    """Translates encoded name into something more readable.

    Args:
        name: Name to translate

    Returns:
        Readable name
    """
    working_str = name
    for k, v in _name_translation_map.items():
        if k in name:
            working_str = working_str.replace(k, v)
    return working_str.replace("_", " ")


class URLConfig(BaseModel):
    """Configuration for URLs associated with an observable."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    inspire_hep: str = Field(..., description="InspireHEP URL")
    hepdata: str = Field(..., description="HEPData URL")
    custom: str | None = Field(None, description="Custom URL")

    @field_validator("inspire_hep", "hepdata", "custom")
    @classmethod
    def validate_urls(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in ["N/A", "TODO"] and not v.startswith("http"):
            msg = "URL must be 'N/A', 'TODO', or start with 'http'"
            raise ValueError(msg)
        return v

    @field_validator("inspire_hep")
    @classmethod
    def validate_inspire_hep(cls, v: str) -> str:
        if "hepdata" in v:
            msg = "inspire_hep link contains hepdata URL - probably transposed"
            raise ValueError(msg)
        return v

    @field_validator("hepdata")
    @classmethod
    def validate_hepdata(cls, v: str) -> str:
        if "inspire" in v:
            msg = "hepdata link contains inspire URL - probably transposed"
            raise ValueError(msg)
        return v


class SoftDropConfig(BaseModel):
    """Configuration for SoftDrop grooming."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    zcut: float = Field(..., description="SoftDrop zcut parameter")
    beta: float = Field(..., description="SoftDrop beta parameter")


class IsolationConfig(BaseModel):
    """Configuration for photon isolation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    type: Literal["full", "charged", "neutral"] = Field(..., description="Isolation type")
    R: Annotated[float, Field(gt=0)] = Field(..., description="Isolation cone radius")
    Et_max: Annotated[float, Field(gt=0)] | None = Field(None, description="Maximum Et for isolation")
    Et_max_pp: Annotated[float, Field(gt=0)] | None = Field(None, description="Maximum Et for pp isolation")
    Et_max_AA: Annotated[float, Field(gt=0)] | None = Field(None, description="Maximum Et for AA isolation")

    @model_validator(mode="after")
    def validate_et_fields(self) -> IsolationConfig:
        # Must have either Et_max or both Et_max_pp and Et_max_AA
        if self.Et_max is None and (self.Et_max_pp is None or self.Et_max_AA is None):
            msg = "Must provide either Et_max or both Et_max_pp and Et_max_AA"
            raise ValueError(msg)
        if self.Et_max is not None and (self.Et_max_pp is not None or self.Et_max_AA is not None):
            msg = "Cannot provide both Et_max and Et_max_{pp,AA}"
            raise ValueError(msg)

        return self


class SmearingConfig(BaseModel):
    """Configuration for gamma smearing."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    detector_level_Et: list[float] = Field(..., description="Detector level Et values")
    particle_level_Et: list[float] = Field(..., description="Particle level Et values")

    @field_validator("detector_level_Et", "particle_level_Et")
    @classmethod
    def validate_et_values(cls, v: list[float]) -> list[float]:
        if not v:
            msg = "Et values cannot be empty"
            raise ValueError(msg)
        if not all(isinstance(val, float) for val in v):
            msg = "All Et values must be floats"
            raise ValueError(msg)
        return v


class TriggerConfig(BaseModel):
    """Configuration for trigger particles."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Momentum fields
    pt: list[float] | None = Field(None, description="Transverse momentum bins")
    pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum transverse momentum")
    Et: list[float] | None = Field(None, description="Transverse energy bins")
    Et_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum transverse energy")

    # Range fields for semi-inclusive measurements
    low_range: list[float] | None = Field(None, description="Low momentum range")
    high_range: list[float] | None = Field(None, description="High momentum range")

    # Eta/rapidity cuts
    eta_cut: float | list[float] | None = Field(None, description="Pseudorapidity cut")
    y_cut: float | None = Field(None, description="Rapidity cut")

    # Specific particle fields
    electron_pt: list[float] | None = Field(None, description="Electron pt bins")
    electron_pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum electron pt")
    electron_eta_cut: float | None = Field(None, description="Electron eta cut")
    electron_y_cut: float | None = Field(None, description="Electron rapidity cut")

    muon_pt: list[float] | None = Field(None, description="Muon pt bins")
    muon_pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum muon pt")
    muon_eta_cut: float | None = Field(None, description="Muon eta cut")
    muon_y_cut: float | None = Field(None, description="Muon rapidity cut")

    z_pt: list[float] | None = Field(None, description="Z boson pt bins")
    z_pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum Z boson pt")
    z_mass: list[float] | None = Field(None, description="Z boson mass range")
    z_y_cut: float | None = Field(None, description="Z boson rapidity cut")

    # Isolation and smearing
    isolation: IsolationConfig | None = Field(None, description="Isolation configuration")
    smearing: SmearingConfig | None = Field(None, description="Smearing configuration")

    @model_validator(mode="after")
    def validate_momentum_fields(self) -> TriggerConfig:
        # Check that we have appropriate momentum fields
        pt_fields = [self.pt, self.pt_min]
        et_fields = [self.Et, self.Et_min]
        range_fields = [self.low_range, self.high_range]

        has_pt = any(field is not None for field in pt_fields)
        has_et = any(field is not None for field in et_fields)
        has_ranges = any(field is not None for field in range_fields)

        # For triggers, we need either pt, Et, or range fields
        if not (has_pt or has_et or has_ranges):
            # Check if we have specific particle fields
            particle_fields = [self.electron_pt_min, self.muon_pt_min, self.z_pt_min]
            has_particle_fields = any(field is not None for field in particle_fields)
            if not has_particle_fields:
                msg = "Must provide momentum fields (pt, Et, or ranges)"
                raise ValueError(msg)

        # Cannot have both pt and Et
        if has_pt and has_et:
            msg = "Cannot provide both pt and Et fields"
            raise ValueError(msg)

        # Validate range fields
        if has_ranges:
            if self.low_range and len(self.low_range) != 2:
                msg = "low_range must have exactly 2 values"
                raise ValueError(msg)
            if self.high_range and len(self.high_range) != 2:
                msg = "high_range must have exactly 2 values"
                raise ValueError(msg)

        return self

    @field_validator("pt", "Et")
    @classmethod
    def validate_momentum_bins(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) < 2:
            msg = "Momentum bins must have at least 2 values"
            raise ValueError(msg)
        return v


class JetConfig(BaseModel):
    """Configuration for jet parameters."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    jet_R: list[float] | float = Field(..., description="Jet radius parameter(s)")
    pt: list[float] | None = Field(None, description="Jet pt range")
    pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum jet pt")
    eta_cut: float | None = Field(None, description="Jet eta cut")
    eta_cut_R: float | None = Field(None, description="Jet eta cut relative to R")
    y_cut: float | None = Field(None, description="Jet rapidity cut")

    @model_validator(mode="after")
    def validate_jet_config(self) -> JetConfig:
        # Must have either pt range or pt_min
        if self.pt is None and self.pt_min is None:
            msg = "Must provide either pt range or pt_min"
            raise ValueError(msg)

        # Must have some eta/rapidity cut
        eta_cuts = [self.eta_cut, self.eta_cut_R, self.y_cut]
        if not any(field is not None for field in eta_cuts):
            msg = "Must provide eta_cut, eta_cut_R, or y_cut"
            raise ValueError(msg)

        return self


class HadronConfig(BaseModel):
    """Configuration for hadron parameters."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pt: list[float] | None = Field(None, description="Hadron pt range")
    pt_min: Annotated[float, Field(gt=0)] | None = Field(None, description="Minimum hadron pt")
    eta_cut: float | None = Field(None, description="Hadron eta cut")
    y_cut: float | None = Field(None, description="Hadron rapidity cut")

    @model_validator(mode="after")
    def validate_hadron_config(self) -> HadronConfig:
        if self.pt is None and self.pt_min is None:
            msg = "Must provide either pt range or pt_min"
            raise ValueError(msg)
        if self.eta_cut is None and self.y_cut is None:
            msg = "Must provide eta_cut or y_cut"
            raise ValueError(msg)
        return self


class BaseObservableConfig(BaseModel):
    """Base configuration for all observables."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(..., description="Whether the observable is enabled")
    centrality: list[tuple[int, int]] = Field(..., description="Centrality bins")
    urls: URLConfig = Field(..., description="URLs for the observable")

    hepdata: str | None = Field(None, description="HEPData file")
    hepdata_pp: str | None = Field(None, description="HEPData file for pp")
    hepdata_AA: str | None = Field(None, description="HEPData file for AA")
    custom_data: str | None = Field(None, description="Custom data file")
    bins: list[float] | None = Field(None, description="Custom binning")

    @field_validator("centrality")
    @classmethod
    def validate_centrality(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        for cent_bin in v:
            if cent_bin[0] >= cent_bin[1]:
                msg = f"Centrality bin {cent_bin} must be in ascending order"
                raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_data_source(self) -> BaseObservableConfig:
        if not any([self.hepdata, self.hepdata_pp, self.hepdata_AA, self.custom_data, self.bins]):
            msg = "Must provide at least one data source (hepdata, custom_data, or bins)"
            raise ValueError(msg)
        return self


class HadronObservableConfig(BaseObservableConfig):
    """Configuration for hadron observables."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    pt: list[float] = Field(..., description="Hadron pt range")
    eta_cut: float | None = Field(None, description="Hadron eta cut")
    y_cut: float | None = Field(None, description="Hadron rapidity cut")

    @field_validator("pt")
    @classmethod
    def validate_pt_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            msg = "pt must have exactly 2 values [min, max]"
            raise ValueError(msg)
        if v[0] >= v[1]:
            msg = "pt range must be in ascending order"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_eta_cut(self) -> HadronObservableConfig:
        if not any(getattr(self, field, None) is not None for field in ["eta_cut", "y_cut"]):
            msg = "Must provide eta_cut or y_cut"
            raise ValueError(msg)
        return self


class HadronCorrelationObservableConfig(BaseObservableConfig):
    """Configuration for hadron correlation observables."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    pt: list[float] = Field(..., description="Hadron pt range")
    trigger_hadron: dict[str, Any] | None = Field(None, description="Trigger hadron config")
    associated_hadron: dict[str, Any] | None = Field(None, description="Associated hadron config")

    @field_validator("pt")
    @classmethod
    def validate_pt_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            msg = "pt must have exactly 2 values [min, max]"
            raise ValueError(msg)
        if v[0] >= v[1]:
            msg = "pt range must be in ascending order"
            raise ValueError(msg)
        return v


class JetObservableConfig(BaseObservableConfig):
    """Configuration for jet observables."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    jet_R: list[float] | float = Field(..., description="Jet radius parameter(s)")
    pt: list[float] = Field(..., description="Jet pt range")
    eta_cut: float | None = Field(None, description="Jet eta cut")
    eta_cut_R: float | None = Field(None, description="Jet eta cut relative to R")
    y_cut: float | None = Field(None, description="Jet rapidity cut")

    # Optional jet-specific parameters
    SoftDrop: list[SoftDropConfig] | None = Field(None, description="SoftDrop configurations")
    alpha: list[float] | None = Field(None, description="Angularity alpha values")
    r: list[float] | None = Field(None, description="Subjet radius values")
    axis: list[dict[str, Any]] | None = Field(None, description="Jet axis configurations")
    kappa: list[float] | None = Field(None, description="Jet charge kappa values")
    track_pt_min: float | None = Field(None, description="Minimum track pt")
    dR: float | None = Field(None, description="Delta R parameter")
    weight: list[float] | None = Field(None, description="EEC weight parameters")
    dynamical_grooming_a: list[float] | None = Field(None, description="Dynamical grooming parameter")

    @field_validator("pt")
    @classmethod
    def validate_pt_range(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            msg = "pt must have exactly 2 values [min, max]"
            raise ValueError(msg)
        if v[0] >= v[1]:
            msg = "pt range must be in ascending order"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_eta_cut(self) -> JetObservableConfig:
        eta_cuts = [self.eta_cut, self.eta_cut_R, self.y_cut]
        if not any(field is not None for field in eta_cuts):
            msg = "Must provide eta_cut, eta_cut_R, or y_cut"
            raise ValueError(msg)
        return self


class TriggeredObservableConfig(BaseObservableConfig):
    """Configuration for triggered observables."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, str_strip_whitespace=True)

    trigger: TriggerConfig = Field(..., description="Trigger configuration")
    jet: JetConfig | None = Field(None, description="Jet configuration")
    hadron: HadronConfig | None = Field(None, description="Hadron configuration")

    # Additional parameters
    dPhi: float | None = Field(None, description="Delta phi requirement")
    c_ref: list[float] | str | None = Field(None, description="Reference cross-section")

    @model_validator(mode="after")
    def validate_recoil_config(self) -> TriggeredObservableConfig:
        # Must have either jet or hadron configuration for recoil
        if self.jet is None and self.hadron is None:
            msg = "Must provide either jet or hadron configuration for recoil"
            raise ValueError(msg)
        return self


class ObservableInfo(BaseModel):
    """Information about an observable extracted from configuration."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True, extra="forbid")

    observable_class: str = Field(..., description="Class of the observable")
    name: str = Field(..., description="Name of the observable")
    config: dict[str, Any] = Field(..., description="Configuration dictionary")

    @property
    def identifier(self) -> str:
        """Unique identifier for the observable."""
        return f"{self.observable_class}_{self.name}"

    @property
    def experiment(self) -> str:
        """Extract experiment name from observable name."""
        return self.name.split("_")[-1].upper()

    def display_name(self) -> str:
        """Pretty print of the observable name."""
        # Remove the experiment name (last part)
        return pretty_print_name("_".join(self.name.split("_")[:-1]))


class ValidationResult(BaseModel):
    """Result of validation process."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    n_observables: int = Field(..., description="Total number of observables")
    n_enabled: int = Field(..., description="Number of enabled observables")
    validation_issues: dict[str, list[str]] = Field(default_factory=dict, description="Validation issues by observable")

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.validation_issues) == 0

    @property
    def enabled_percentage(self) -> float:
        """Percentage of enabled observables."""
        if self.n_observables == 0:
            return 0.0
        return (self.n_enabled / self.n_observables) * 100


def extract_observables(config: dict[str, Any]) -> dict[str, ObservableInfo]:
    """Extract observable info from the configuration file.

    Args:
        config: YAML analysis config.
    Returns:
        Observable info extracted from the configuration file.
    """
    # Determine which keys are valid classes of observables
    observable_classes = []
    for k in config:
        # By convention, we always start observables with the simple hadron class
        if "hadron" in k or observable_classes:
            observable_classes.append(k)

    observables = {}
    for observable_class in observable_classes:
        if not isinstance(config[observable_class], dict):
            continue

        for observable_key in config[observable_class]:
            observable_info = config[observable_class][observable_key]
            observables[f"{observable_class}_{observable_key}"] = ObservableInfo(
                observable_class=observable_class,
                name=observable_key,
                config=observable_info,
            )

    return observables


def validate_observable_with_pydantic(observable_info: ObservableInfo) -> list[str]:
    """Validate a single observable using Pydantic models.

    Args:
        observable_info: Observable information to validate

    Returns:
        List of validation issues
    """
    issues = []
    config = observable_info.config
    observable_class = observable_info.observable_class

    try:
        # Choose the appropriate Pydantic model based on observable class
        if observable_class == "hadron":
            HadronObservableConfig(**config)
        elif observable_class == "hadron_correlation":
            HadronCorrelationObservableConfig(**config)
        elif observable_class.startswith("inclusive") and "jet" in observable_class:
            JetObservableConfig(**config)
        elif "trigger" in observable_class:
            TriggeredObservableConfig(**config)
        else:
            # Fall back to base observable config
            BaseObservableConfig(**config)

    except ValidationError as e:
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            issues.append(f"{field_path}: {error['msg']}")

    return issues


def validate_observables(observables: dict[str, ObservableInfo]) -> ValidationResult:
    """Validate the observable configuration using Pydantic models.

    Args:
        observables: Observable info extracted from an analysis config.
    Returns:
        Validation result with issues by observable.
    """
    validation_issues = defaultdict(list)
    n_observables = 0
    n_enabled = 0

    for key, observable_info in observables.items():
        logger.debug(f"Checking {key}")
        n_observables += 1

        # Check if enabled
        if observable_info.config.get("enabled", False):
            n_enabled += 1

        # Validate using Pydantic
        pydantic_issues = validate_observable_with_pydantic(observable_info)
        if pydantic_issues:
            validation_issues[key].extend(pydantic_issues)

    return ValidationResult(n_observables=n_observables, n_enabled=n_enabled, validation_issues=dict(validation_issues))


def validate_yaml(filename: Path) -> ValidationResult:
    """Driver function for validating an analysis config.

    Args:
        filename: Path to the analysis config.
    Returns:
        Validation result.
    """
    logger.info(f"Validating {filename}...")

    with filename.open() as f:
        config = yaml.safe_load(f)

    # First extract all of the observables from the config
    observables = extract_observables(config=config)

    # And then validate the conditions
    return validate_observables(observables=observables)


def validate_yaml_entry_point() -> None:
    """Entry point for validation of the analysis config."""
    # Setup
    helpers.setup_logging(level=logging.DEBUG)

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Validate a given analysis config",
    )
    parser.add_argument(
        "analysis_config",
        help="Analysis config file to validate",
        type=Path,
    )

    args = parser.parse_args()

    result = validate_yaml(
        filename=args.analysis_config,
    )

    logger.info(
        f"Summary: # observables: {result.n_observables}, "
        f"# enabled: {result.n_enabled} -> {result.enabled_percentage:.2f}% enabled"
    )

    if result.is_valid:
        logger.info("🎉 Validation success!")
    else:
        logger.error("Validation issues:")
        helpers.rich_console.print(result.validation_issues)
        logger.error("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    validate_yaml_entry_point()
