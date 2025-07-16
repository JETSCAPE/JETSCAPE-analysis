"""Validation for analysis yaml file.

This isn't comprehensive - it's just what needs to be minimally supported.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import argparse
import logging
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
            # if "v2" in observable_key and observable_class != "dijet":
            if False:
                # need to go a level deeper for the v2 since they're nested...
                for sub_observable_key in config[observable_class][observable_key]:
                    observable_info = config[observable_class][observable_key][sub_observable_key]

                    # Move the experiment to the end of the name to match the convention
                    *base_observable_name, experiment_name = observable_key.split("_")
                    observable_name = "_".join(base_observable_name)
                    observable_name += f"_{sub_observable_key}_{experiment_name}"

                    observables[f"{observable_class}_{observable_key}_{sub_observable_key}"] = ObservableInfo(
                        observable_class=observable_class,
                        name=observable_name,
                        config=observable_info,
                    )
            else:
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

    for key, observable_info in observables.items():
        config = observable_info.config
        if "enabled" not in config:
            validation_issues[key].append("Missing 'enabled' key")
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

        if "jet" in observable_info.observable_class and "jet_R" not in observable_info.config:
            validation_issues[key].append("Missing jet_R")
        if not any(v in config for v in ["eta_cut", "eta_cut_R", "y_cut", "eta_cut_jet", "eta_cut_hadron"]):
            # eta_cut_jet and eta_cut_hadron is for when we need to specify between the trigger and the recoil
            validation_issues[key].append(
                "Missing eta_cut, eta_cut_R, y_cut, eta_cut_jet, or eta_cut_hadron (as appropriate for the observable)"
            )

    # Convert to standard dict just to avoid confusion
    return dict(validation_issues)


def validate_yaml(filename: Path) -> dict[str, list]:
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

    res = validate_yaml(
        filename=args.analysis_config,
    )
    if not res:
        logger.info("🎉 Success!")
    else:
        logger.error("Validation issues:")
        helpers.rich_console.print(res)
        logger.error("❌ Validation failed!")


if __name__ == "__main__":
    validate_yaml_entry_point()
