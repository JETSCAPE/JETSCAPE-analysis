"""Validation for analysis yaml file.

This isn't comprehensive - it's just what needs to be minimally supported.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import attrs
import yaml

from jetscape_analysis.base import helpers

logger = logging.getLogger(__name__)


# Translate names as needed
_name_translation_map = {}


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
class ObservableInfo:
    observable_class: str = attrs.field()
    name: str = attrs.field()
    config: dict[str, Any] = attrs.field()

    @property
    def identifier(self) -> str:
        return f"{self.observable_class}_{self.name}"

    @property
    def experiment(self) -> str:
        return self.name.split("_")[-1].upper()

    def display_name(self) -> str:
        """Pretty print of the observable name"""
        # -1 removes the experiment name
        return pretty_print_name("_".join(self.name.split("_")[:-1]))


@attrs.frozen
class Result:
    n_observables: int = attrs.field()
    n_enabled: int = attrs.field()
    validation_issues: dict[str, list[str]] = attrs.Factory(dict)


def extract_observables(config: dict[str, Any]) -> dict[str, ObservableInfo]:
    """Extract observable info from the configuration file.

    Args:
        config: YAML analysis config.
    Returns:
        Observable info extracted from the configuration file.
    """
    # Need to determine which keys are valid classes of observables.
    observable_classes = []
    for k in config:
        # By convention, we always start observables with the simple hadron class
        if "hadron" in k or observable_classes:
            observable_classes.append(k)

    observables = {}
    for observable_class in observable_classes:
        for observable_key in config[observable_class]:
            observable_info = config[observable_class][observable_key]
            observables[f"{observable_class}_{observable_key}"] = ObservableInfo(
                observable_class=observable_class,
                name=observable_key,
                config=observable_info,
            )

    return observables


def validate_observables(observables: dict[str, ObservableInfo]) -> dict[str, list]:  # noqa: C901
    """Validate the observable configuration.

    NOTE:
        This **IS NOT** a full specification. Validation checks are just added as
        they come up.

    Args:
        observables: Observable info extracted from an analysis config.
    Returns:
        List of validation issues by observable.
    """
    # Setup
    validation_issues = defaultdict(list)
    n_observables = 0
    n_enabled = 0

    for key, observable_info in observables.items():
        logger.debug(f"Checking {key}")

        n_observables += 1
        config = observable_info.config
        # Required fields
        # Enabled field
        if "enabled" not in config:
            validation_issues[key].append("Missing 'enabled' key")
        elif config["enabled"]:
            # Also keep track of what is actually enabled for statistics
            n_enabled += 1
        # Centrality field
        centrality = config.get("centrality", [])
        if not centrality:
            validation_issues[key].append("Missing 'centrality' key")
        else:
            # logger.debug(f"Checking centrality: {centrality}")
            # Must be a list of lists
            for cent in centrality:
                centrality_issues = []
                if len(cent) != 2:
                    centrality_issues.append(f"{cent} is not a pair of two values")
                elif not (isinstance(cent[0], int) and isinstance(cent[1], int)):
                    centrality_issues.append(
                        f"{cent} should be pair of ints, but types are {type(cent[0])}, {type(cent[1])}"
                    )

            if centrality_issues:
                validation_issues[key].extend(centrality_issues)
        # HEPdata or custom_data field
        if not any(v in ["hepdata", "hepdata_pp", "hepdata_AA", "custom_data", "bins"] for v in observable_info.config):
            validation_issues[key].append("Missing required hepdata, custom_data, or binning field!")

        if "urls" not in config:
            validation_issues[key].append("Missing 'urls'")
        else:
            required_url_keys = ["inspire_hep", "hepdata"]
            possible_url_keys = ["custom"]
            all_possible_url_keys = required_url_keys + possible_url_keys

            urls = config["urls"]
            unexpected_keys = [v for v in urls if v not in all_possible_url_keys]

            # Unexpected keys
            if any(unexpected_keys):
                validation_issues[key].append(f"Unexpected URL key: {unexpected_keys}")

            # Required keys
            if not all(v in urls for v in required_url_keys):
                validation_issues[key].append(
                    f"Missing required URL key: {required_url_keys=}, provided: {list(urls.keys())}"
                )

            # Validate what's stored. Needs to be either: N/A, TODO, or start with "http"
            invalid_urls = {}
            for k, v in urls.items():
                if v not in ["N/A", "TODO"] and not v.startswith("http"):
                    invalid_urls[k] = v
            if invalid_urls:
                validation_issues[key].append(f"Invalid URLs. Must be N/A or a URL. Problematic value: {invalid_urls}")

            # Check we haven't transposed URLs (i.e. stored hepdata in inspire_hep or vise-versa)
            if "hepdata" in urls["inspire_hep"]:
                validation_issues[key].append(
                    f"inspire_hep link contains hepdata url. Probably transposed? {urls['inspire_hep']=}"
                )

            if "inspire" in urls["hepdata"]:
                validation_issues[key].append(
                    f"hepdata link contains inspire url. Probably transposed? {urls['hepdata']=}"
                )

        # Further checks are based on the type of the analysis
        # Inclusive observables - inclusive hadron, hadron correlations, or inclusive jet
        if observable_info.observable_class.startswith("inclusive") or observable_info.observable_class == "hadron":
            issues = []
            # Jet properties
            if "jet" in observable_info.observable_class:
                issues.extend(_check_jet_properties(config=observable_info.config))
            # Hadron properties
            if "hadron" in observable_info.observable_class:
                issues.extend(_check_hadron_properties(config=observable_info.config))

            if issues:
                validation_issues[key].extend(issues)

        if observable_info.observable_class == "hadron_correlation":
            issues = []
            # Hadron correlation properties
            issues.extend(_check_hadron_correlation_properties(config=observable_info.config))

            if issues:
                validation_issues[key].extend(issues)

        # Triggered observables
        if "trigger" in observable_info.observable_class:
            # Trigger properties
            issues = []
            trigger_to_validation_function = {
                "hadron": _check_hadron_trigger_properties,
                "dijet": _check_dijet_trigger_properties,
                "pion": _check_pion_trigger_properties,
                "gamma": _check_gamma_trigger_properties,
                "z": _check_z_trigger_properties,
            }
            for trigger_name, func in trigger_to_validation_function.items():
                if f"{trigger_name}_trigger" in observable_info.observable_class:
                    # The config will always be named "trigger" regardless of the type, so we can
                    # always ask for the same name here.
                    res = func(config=observable_info.config.get("trigger", {}))
                    # Add explicit labels for clarity
                    issues.extend([f"{trigger_name}_trigger: {v}" for v in res])

            # Recoil/associated properties
            # NOTE: The observable_class are of the form "X_trigger_Y", where X is the trigger and Y is the recoil/associated.
            #       Since we only want to target the recoil here, we look for "_trigger_Y" for Y
            # Jet properties
            # NOTE: Need to explicitly check for chjet and jet since we're including more of the observable class
            if (
                "_trigger_chjet" in observable_info.observable_class
                or "_trigger_jet" in observable_info.observable_class
            ):
                issues.extend(_check_jet_properties(config=observable_info.config.get("jet", {})))
            # Hadron properties
            if "_trigger_hadron" in observable_info.observable_class:
                issues.extend(_check_hadron_properties(config=observable_info.config.get("hadron", {})))

            if issues:
                validation_issues[key].extend(issues)

    # Convert to standard dict just to avoid confusion
    return Result(n_observables=n_observables, n_enabled=n_enabled, validation_issues=dict(validation_issues))


def _check_hadron_trigger_properties(config: dict[str, Any]) -> list[str]:
    """Check hadron trigger properties in config.

    Args:
        config: Configuration containing the parameters relevant to hadron triggers.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking hadron trigger properties")

    issues = []
    if not config:
        issues.append("Missing trigger config!")
    # Use the existing hadron configuration properties
    issues.extend(_check_hadron_properties_impl(config=config, trigger=True))
    return issues


def _check_dijet_trigger_properties(config: dict[str, Any]) -> list[str]:
    """Check dijet trigger properties in config.

    Args:
        config: Configuration containing the parameters relevant to dijet triggers.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking dijet trigger properties")

    issues = []
    if not config:
        issues.append("Missing trigger config!")
    # Use the existing jet configuration properties
    issues.extend(_check_jet_properties_impl(config=config))
    return issues


def _check_pion_trigger_properties(config: dict[str, Any]) -> list[str]:
    """Check pion trigger properties in config.

    Args:
        config: Configuration containing the parameters relevant to pion triggers.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking pion trigger properties")

    issues = []
    if not config:
        issues.append("Missing trigger config!")

    # Need either "pt" or "Et"
    momentum_fields = ["pt", "Et"]
    available_momentum_fields = [v for v in momentum_fields if v in config]
    if len(available_momentum_fields) != 1:
        issues.append(f"Wrong number of momentum fields. Must include one of: {momentum_fields}")
    for k in available_momentum_fields:
        value = config[k]
        if "min" in k and not isinstance(value, float):
            issues.append(f"`{k}` field not formatted correctly. Needs a single value, provided: {value=}")
        elif len(value) < 2:
            issues.append(f"`{k}` field not formatted correctly. Needs at least two values, provided: {value=}")
    # Check for lower case "et" (which is a misspelling of Et)
    if any(v in config for v in ["et", "et_min"]):
        issues.append(
            f"Contains `et` or `et_min`, which should be spelled `Et` or `Et_min`. Provided keys: {config.keys()}"
        )

    # Eta requirement
    if not any(v in config for v in ["eta_cut", "y_cut"]):
        issues.append(f"Missing eta_cut or y_cut (as appropriate for the observable). Provided keys: {config.keys()}")

    return issues


def _check_gamma_trigger_properties(config: dict[str, Any]) -> list[str]:
    """Check gamma trigger properties in config.

    Args:
        config: Configuration containing the parameters relevant to gamma triggers.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking gamma trigger properties")

    issues = []
    if not config:
        issues.append("Missing trigger config!")

    # Need either "pt" or "Et"
    pt_issues = _check_standard_momentum_field(config=config)
    Et_issues = _check_standard_momentum_field(config=config, check_Et_instead=True)
    if pt_issues and Et_issues:
        issues.append(
            "Need to provide either pt or Et fields, but detected issues with both. Please check the other reported issues for details."
        )
        # NOTE: Would use a tab for spacing, but it doesn't evaluate nicely in our printout, so just doing it by hand.
        issues.extend([f"    {v}" for v in pt_issues])
        issues.extend([f"    {v}" for v in Et_issues])
    # Check for lower case "et" (which is a misspelling of Et)
    if any(v in config for v in ["et", "et_min"]):
        issues.append(
            f"Contains `et` or `et_min`, which should be spelled `Et` or `Et_min`. Provided keys: {config.keys()}"
        )

    # Eta requirement
    if not any(v in config for v in ["eta_cut", "y_cut"]):
        issues.append(f"Missing eta_cut or y_cut (as appropriate for the observable). Provided keys: {config.keys()}")

    # Isolation or smearing properties
    # Need an isolation type
    isolation_config = config.get("isolation")
    smearing_config = config.get("smearing")
    if not isolation_config and not smearing_config:
        issues.append("Must provide an isolation or smearing configuration.")
    elif isolation_config:
        issues.extend(_isolation_validation(isolation_config=isolation_config))
    elif smearing_config:
        issues.extend(_gamma_smearing_validation(smearing_config=smearing_config))

    return issues


def _isolation_validation(isolation_config: dict[str, Any]) -> list[str]:
    """Isolation validation

    Args:
        isolation_config: Configuration containing the parameters relevant to the isolation parameters.
    Returns:
        Any issues observed with the configuration.
    """
    issues = []
    # Type
    isolation_type = isolation_config.get("type")
    if not isolation_type:
        issues.append("Missing required isolation")
    elif isolation_type not in ["full", "charged", "neutral"]:
        issues.append(
            f"Invalid isolation type {isolation_type}. Isolation type must be 'full', 'charged', or 'neutral'."
        )
    # Cone size
    if "R" not in isolation_config:
        issues.append("Missing isolation R")
    # Et_max requirements
    # Should be of the form Et_max or (Et_max_pp and Et_max_AA)
    Et_fields = [f for f in isolation_config if "Et" in f]
    if not Et_fields:
        issues.append("Missing required isolation Et fields. Please provide Et_max or Et_max_{pp,AA}")
    elif len(Et_fields) > 2:
        issues.append(f"Too many isolation Et fields. Found: {Et_fields}")
    elif len(Et_fields) == 2:
        # Verify the types
        incorrect_types = {
            field: isolation_config[field] for field in Et_fields if not isinstance(isolation_config[field], float)
        }
        if incorrect_types:
            issues.append(f"Isolation `Et_max` values should be float. Incorrect value -> index map: {incorrect_types}")
    elif len(Et_fields) == 1 and not isinstance(isolation_config["Et_max"], float):
        issues.append("Isolation `Et_max` value should be float.")

    return issues


def _gamma_smearing_validation(smearing_config: dict[str, Any]) -> list[str]:
    """Gamma smearing configuration validation

    Args:
        smearing_config: Configuration containing the parameters relevant to the smearing parameters.
    Returns:
        Any issues observed with the configuration.
    """
    issues = []
    fields = ["detector_level_Et", "particle_level_Et"]
    for field in fields:
        values = smearing_config.get(field)
        if not values:
            issues.append(f"Missing values for {field}")
            continue

        # Verify the types
        incorrect_types = {value: i for i, value in enumerate(values) if not isinstance(value, float)}
        if incorrect_types:
            issues.append(
                f"Gamma smearing field `{field}` values should be float. Incorrect value -> index map: {incorrect_types}"
            )

    return issues


def _check_standard_momentum_field(
    config: dict[str, Any], prefix: str = "", check_Et_instead: bool = False
) -> list[str]:
    """Check that a standard momentum field is formatted properly.

    Args:
        config: Configuration containing the momentum field(s).
        prefix: Prefix to be used when accessing the fields. e.g. if prefix is `muon`,
            then it would check `muon_pt`.
        check_Et_instead: Instead of checking for pt, check for Et based fields
            (generally this is used for photon triggers).
    Returns:
        List of issues associated with the momentum fields.
    """
    issues = []
    # Setup
    pt_field_name = "pt"
    if check_Et_instead:
        pt_field_name = "Et"

    # Assign prefix as needed
    if prefix:
        pt_field_name = f"{prefix}_{pt_field_name}"

    # If it exists, check the formatting
    pt = config.get(pt_field_name)
    if pt:
        if len(pt) < 2:
            issues.append(
                f"`{pt_field_name}` field not formatted correctly. Needs at least two values, provided: {pt=}"
            )
        # Account for the min_pt case, where the last value is null rather than being set.
        treat_as_min_pt = pt[-1] is None
        ending_index = -1 if treat_as_min_pt else None

        # Check for incorrectly typed values. Should be float.
        wrong_type_index = [i for i, value in enumerate(pt[:ending_index]) if not isinstance(value, float)]
        if wrong_type_index:
            logger.info(f"{wrong_type_index=}, {pt=}")
            values = {pt[i]: i for i in wrong_type_index}
            issues.append(f"`{pt_field_name}` values should be float. Incorrect value -> index map: {values}")
    else:
        issues.append(f"Need field `{pt_field_name}`")

    return issues


def _check_z_trigger_properties(config: dict[str, Any]) -> list[str]:
    """Check z trigger properties in config.

    Args:
        config: Configuration containing the parameters relevant to z triggers.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking Z boson trigger properties")

    issues = []
    if not config:
        issues.append("Missing trigger config!")

    # Electron requirements
    # They're not always analyzed, but if they're included, they should be formatted correctly
    electron_included = any("electron" in k for k in config)
    if electron_included:
        issues.extend(_check_standard_momentum_field(config=config, prefix="electron"))
        # Eta requirement
        if not any(v in config for v in ["electron_eta_cut", "electron_y_cut"]):
            issues.append(
                f"Missing eta_cut or y_cut (as appropriate for the observable). Provided keys: {config.keys()}"
            )
    # Muon requirements
    # Muons always seem to be used, so we always check them.
    issues.extend(_check_standard_momentum_field(config=config, prefix="muon"))
    # Eta requirement
    if not any(v in config for v in ["muon_eta_cut", "muon_y_cut"]):
        issues.append(f"Missing eta_cut or y_cut (as appropriate for the observable). Provided keys: {config.keys()}")
    # z requirements
    # mass
    if "z_mass" not in config:
        issues.append("Missing z mass selection field.")
    # pt
    issues.extend(_check_standard_momentum_field(config=config, prefix="z"))
    # eta selections on Z are often omitted, so we leave them off here.

    return issues


def _check_hadron_correlation_properties(config: dict[str, Any]) -> list[str]:
    """Check hadron correlation properties in config.

    Args:
        config: Configuration containing the parameters relevant to hadron correlations.
            Could be the main config, but could also be the more specific config
            for a triggered observable.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking hadron correlation properties")

    issues = []
    if not config:
        issues.append("Missing hadron correlation config!")

    # Need either "pt" or "pt_min"
    issues.extend(_check_standard_momentum_field(config=config))

    return issues


def _check_jet_properties(config: dict[str, Any]) -> list[str]:
    """Check jet properties in config.

    Separated from the implementation for clarity of logging where the call originates.

    Args:
        config: Configuration containing the parameters relevant to jets.
            Could be the main config, but could also be e.g. the recoil jet config
            for a triggered observable.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking jet properties")
    return _check_jet_properties_impl(config=config)


def _check_jet_properties_impl(config: dict[str, Any]) -> list[str]:
    """Check jet properties in config implementation.

    Args:
        config: Configuration containing the parameters relevant to jets.
            Could be the main config, but could also be e.g. the recoil jet config
            for a triggered observable.
    Returns:
        Any issues observed with the configuration.
    """
    issues = []
    if not config:
        issues.append("Missing jet config!")
    if "jet_R" not in config:
        issues.append("Missing jet_R")
    # Need either "pt" or "pt_min"
    issues.extend(_check_standard_momentum_field(config=config))
    # Eta requirement
    if not any(v in config for v in ["eta_cut", "eta_cut_R", "y_cut"]):
        issues.append(
            f"Missing eta_cut, eta_cut_R, y_cut, eta_cut_jet (as appropriate for the observable). Provided keys: {config.keys()}"
        )

    # Label each issue for clarity
    return [f"jet: {v}" for v in issues]


def _check_hadron_properties(config: dict[str, Any]) -> list[str]:
    """Check hadron properties in config.

    Separated from the implementation for clarity of logging where the call originates.

    Args:
        config: Configuration containing the parameters relevant to hadrons.
            Could be the main config, but could also be e.g. the recoil hadron config
            for a triggered observable.
    Returns:
        Any issues observed with the configuration.
    """
    logger.debug("-> Checking hadron properties")
    return _check_hadron_properties_impl(config=config)


def _check_hadron_properties_impl(config: dict[str, Any], trigger: bool = False) -> list[str]:
    """Check hadron properties in config implementation.

    Args:
        config: Configuration containing the parameters relevant to hadrons.
            Could be the main config, but could also be e.g. the recoil hadron config
            for a triggered observable.
    Returns:
        Any issues observed with the configuration.
    """
    issues = []
    if not config:
        issues.append("Missing hadron config!")

    # Momentum field
    check_for_standard_momentum_fields = True
    if trigger:
        # Look for low + high range fields instead of the standard.
        # This tells us it's a semi-inclusive measurement.
        trigger_ranges = [v for v in config if "_range" in v]
        if trigger_ranges:
            check_for_standard_momentum_fields = False
            # Validate the values.
            for trigger_range in trigger_ranges:
                trigger_range_values = config[trigger_range]
                if len(trigger_range_values) != 2:
                    issues.append(
                        f"Wrong specification of trigger range {trigger_range}. Need two values, but provided: {trigger_range_values}"
                    )
                incorrect_types = {
                    i: value for i, value in enumerate(config[trigger_range]) if not isinstance(value, float)
                }
                if incorrect_types:
                    issues.append(
                        f"Field `{trigger_range}` values should be float. Incorrect value -> index map: {incorrect_types}"
                    )
    # We will always do this check, as long as we don't have the special trigger case handled above.
    if check_for_standard_momentum_fields:
        issues.extend(_check_standard_momentum_field(config=config))

    # Eta requirement
    if not any(v in config for v in ["eta_cut", "y_cut"]):
        issues.append(f"Missing eta_cut or y_cut (as appropriate for the observable). Provided keys: {config.keys()}")

    # Label each issue for clarity
    return [f"hadron: {v}" for v in issues]


def validate_yaml(filename: Path) -> Result:
    """Driver function for validating an analysis config.

    Args:
        filename: Path to the analysis config.
    Returns:
        List of validation issues by observable.
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
        f"Summary:    # observables: {result.n_observables}, # enabled: {result.n_enabled} -> {result.n_enabled / result.n_observables * 100:.2f}% enabled"
    )
    if not result.validation_issues:
        logger.info("🎉 Validation success!")
    else:
        logger.error("Validation issues:")
        helpers.rich_console.print(result.validation_issues)
        logger.error("❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    validate_yaml_entry_point()
