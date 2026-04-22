from __future__ import annotations

import logging
from typing import Any

import pytest
import ruamel.yaml
from jetscape_analysis.data_curation import observable

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "spec",
    [
        observable.CentralitySpec(0.0, 10.0),
        observable.PtSpec(10.0, 20.0),
        observable.EtaSpec(1.0),
        observable.EtaSpec(min=0.3, max=1.0),
        observable.RapiditySpec(1.0),
        observable.MassSpec(70.0, 120.0),
        observable.JetRSpec(0.4),
        observable.GroomingSettingsSpec(observable.SoftDropSpec(0.2, 0.1)),
        observable.GroomingSettingsSpec(method={"type": "soft_drop", "z_cut": 0.1, "beta": 0.5}),
        observable.GroomingSettingsSpec(observable.DynamicalGroomingSpec(1.0)),
        observable.GroomingSettingsSpec(method={"type": "dynamical_grooming", "a": 2.0}),
        observable.JetAxisDifferenceSpec("WTA_SD", observable.SoftDropSpec(0.2, 0.1)),
        observable.JetAxisDifferenceSpec("WTA_SD", grooming_settings={"type": "soft_drop", "z_cut": 0.2, "beta": 0.5}),
        observable.JetAxisDifferenceSpec("WTA_Standard"),
        observable.AngularitySpec(2.0),
        observable.AngularitySpec(2.0, 1.0),
        observable.JetChargeSpec(0.3),
        observable.SubjetRSpec(0.1),
        observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0)),
        observable.SmearingSpec(observable.EtSpec(9.0, 22.0), observable.EtSpec(10.0, 15.0)),
        observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 5.0), observable.EtSpec(0, 10.0)),
    ],
    ids=lambda x: f"{x!r}",
)
def test_param_spec_round_trip(spec: observable.ParameterSpec, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    encoded = spec.encode()
    decoded = spec.decode(encoded)
    assert decoded == spec

    # For testing as needed...
    # if isinstance(spec, observable.GroomingSettingsSpec):
    #     pytest.fail("Need to inspect")


@pytest.mark.parametrize(
    "specs",
    [
        # Options that we consider for all specs:
        # 1. Standard single value
        # 2. Multiple values
        # 3. Multiple values with label (as appropriate)
        # Centrality:
        observable.CentralitySpecs(values=[observable.CentralitySpec(0.0, 10.0)]),
        observable.CentralitySpecs(
            values=[observable.CentralitySpec(0.0, 10.0), observable.CentralitySpec(20.0, 40.0)]
        ),
        # No label needed for centrality
        # Pt:
        observable.PtSpecs(values=[observable.PtSpec(20.0, 30.0)]),
        observable.PtSpecs(values=[observable.PtSpec(20.0, 30.0), observable.PtSpec(30.0, 50.0)]),
        observable.PtSpecs(values=[observable.PtSpec(20.0, 30.0), observable.PtSpec(30.0, 50.0)], label="jet"),
        # Eta
        observable.EtaSpecs(values=[observable.EtaSpec(1.0)]),
        observable.EtaSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)]),
        observable.EtaSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)], label="jet"),
        # Eta range
        observable.EtaSpecs(
            values=[observable.EtaSpec(min=0.3, max=1.0), observable.EtaSpec(min=0.2, max=1.2)], label="jet"
        ),
        # Eta_R
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0)]),
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)]),
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)], label="jet"),
        # Eta range
        observable.EtaRSpecs(
            values=[observable.EtaSpec(min=0.3, max=1.0), observable.EtaSpec(min=0.2, max=1.2)], label="jet"
        ),
        # Rapidity
        observable.RapiditySpecs(values=[observable.RapiditySpec(1.0)]),
        observable.RapiditySpecs(values=[observable.RapiditySpec(1.0), observable.RapiditySpec(1.2)]),
        observable.RapiditySpecs(values=[observable.RapiditySpec(1.0), observable.RapiditySpec(1.2)], label="jet"),
        # Mass
        observable.MassSpecs(values=[observable.MassSpec(70.0, 120.0)]),
        observable.MassSpecs(values=[observable.MassSpec(70.0, 120.0), observable.MassSpec(90.0, 110.0)]),
        observable.MassSpecs(values=[observable.MassSpec(70.0, 120.0), observable.MassSpec(90.0, 110.0)], label="jet"),
        # Jet R
        observable.JetRSpecs(values=[observable.JetRSpec(0.2)]),
        observable.JetRSpecs(values=[observable.JetRSpec(0.2), observable.JetRSpec(0.4), observable.JetRSpec(0.6)]),
        observable.JetRSpecs(values=[observable.JetRSpec(0.2), observable.JetRSpec(0.4)], label="jet"),
        # Grooming settings
        ## SD
        observable.GroomingSettingsSpecs(values=[observable.GroomingSettingsSpec(observable.SoftDropSpec(0.2, 0.1))]),
        observable.GroomingSettingsSpecs(
            values=[
                observable.GroomingSettingsSpec(observable.SoftDropSpec(0.2, 0.1)),
                observable.GroomingSettingsSpec(observable.SoftDropSpec(0.1, 0.0)),
            ]
        ),
        observable.GroomingSettingsSpecs(
            values=[
                observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.2, "beta": 0.1}),
                observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.1, "beta": 0.0}),
            ],
            label="jet",
        ),
        ## DyG
        observable.GroomingSettingsSpecs(
            values=[
                observable.GroomingSettingsSpec({"type": "dynamical_grooming", "a": 1.0}),
                observable.GroomingSettingsSpec({"type": "dynamical_grooming", "a": 2.0}),
            ]
        ),
        ## Mixed values
        # NOTE: Just trying the two methods here, but either could be used to construct the GroomingSettingsSpec. The dict happens to be more concise.
        observable.GroomingSettingsSpecs(
            values=[
                observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.2, "beta": 0.1}),
                observable.GroomingSettingsSpec(observable.DynamicalGroomingSpec(2.0)),
            ],
            label="jet",
        ),
        # Jet-axis difference
        observable.JetAxisDifferenceSpecs(
            values=[
                observable.JetAxisDifferenceSpec(
                    "WTA_SD", observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.2, "beta": 0.1})
                )
            ]
        ),
        observable.JetAxisDifferenceSpecs(
            values=[
                observable.JetAxisDifferenceSpec(
                    "WTA_SD", observable.GroomingSettingsSpec(observable.SoftDropSpec(0.2, 0.1))
                ),
                observable.JetAxisDifferenceSpec(
                    "WTA_SD", observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.2, "beta": 0.0})
                ),
            ]
        ),
        observable.JetAxisDifferenceSpecs(
            values=[
                observable.JetAxisDifferenceSpec(
                    "WTA_SD", observable.GroomingSettingsSpec(observable.SoftDropSpec(0.2, 0.1))
                ),
                observable.JetAxisDifferenceSpec(
                    "WTA_SD", observable.GroomingSettingsSpec({"type": "soft_drop", "z_cut": 0.2, "beta": 0.0})
                ),
                observable.JetAxisDifferenceSpec("WTA_Standard"),
            ],
            label="jet",
        ),
        # Angularities
        observable.AngularitySpecs(values=[observable.AngularitySpec(1.0)]),
        observable.AngularitySpecs(values=[observable.AngularitySpec(1.0), observable.AngularitySpec(2.0)]),
        observable.AngularitySpecs(
            values=[observable.AngularitySpec(1.0, 1.1), observable.AngularitySpec(2.0, 1.1)], label="jet"
        ),
        # Jet charge
        observable.JetChargeSpecs(values=[observable.JetChargeSpec(1.0)]),
        observable.JetChargeSpecs(values=[observable.JetChargeSpec(1.0), observable.JetChargeSpec(2.0)]),
        observable.JetChargeSpecs(values=[observable.JetChargeSpec(1.0), observable.JetChargeSpec(2.0)], label="jet"),
        # Subjet R
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0)]),
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0), observable.SubjetRSpec(2.0)]),
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0), observable.SubjetRSpec(2.0)], label="jet"),
        # Smearing
        observable.SmearingSpecs(
            values=[observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0))]
        ),
        observable.SmearingSpecs(
            values=[
                observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0)),
                observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0)),
            ]
        ),
        observable.SmearingSpecs(
            values=[
                observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0)),
                observable.SmearingSpec(observable.PtSpec(9.0, 22.0), observable.PtSpec(10.0, 15.0)),
            ],
            label="pion_trigger",
        ),
        observable.SmearingSpecs(
            values=[
                observable.SmearingSpec(observable.EtSpec(9.0, 22.0), observable.EtSpec(10.0, 15.0)),
                observable.SmearingSpec(observable.EtSpec(9.0, 22.0), observable.EtSpec(10.0, 15.0)),
            ],
            label="pion_trigger",
        ),
        # Isolation
        observable.IsolationSpecs(
            values=[observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 5.0), observable.EtSpec(0, 10.0))],
        ),
        observable.IsolationSpecs(
            values=[
                observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 5.0), observable.EtSpec(0, 6.0)),
                observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 6.0), observable.EtSpec(0, 7.0)),
            ],
        ),
        observable.IsolationSpecs(
            values=[
                observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 5.0), observable.EtSpec(0, 6.0)),
                observable.IsolationSpec("neutral", observable.JetRSpec(0.4), observable.EtSpec(0, 6.0), observable.EtSpec(0, 7.0)),
            ],
            label="gamma_trigger",
        ),
    ],
    ids=lambda x: f"{x!r}",
)
def test_param_specs_round_trip(specs: observable.ParameterSpecs, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    encoded = specs.encode()
    decoded = specs.decode(encoded)
    assert decoded == specs

    # For testing as needed...
    # if isinstance(specs, observable.JetAxisDifferenceSpecs):
    #     pytest.fail("Need to inspect")


@pytest.mark.parametrize("add_label", [False, True])
@pytest.mark.parametrize(
    "inputs",
    [
        # Centrality
        # Since it's the only "base" property, it routes directly to the from_config rather than
        # through an intermediate function.
        (
            observable.CentralitySpecs.from_config,
            """
centrality: [[0, 10], [20, 30], [40, 50]]
""",
            "",
        ),
        # Hadron parameters
        (
            observable.extract_hadron_parameters,
            """
pt: [20., 40.]
eta: 0.9
""",
            "hadron",
        ),
        # Jet parameters
        (
            observable.extract_jet_parameters,
            """
R: [0.2]
pt: [60., 80.]
eta_R: 0.9
soft_drop:
  - { "z_cut": 0.2, "beta": 0 }
dynamical_grooming:
  - { "a": 1.0 }
""",
            "jet",
        ),
        # Pion parameters
        (
            observable.extract_pion_parameters,
            """
  eta: 1.0
  Et: [9., 22.]
  smearing:
    - {detector_level_Et: [11., 15.], particle_level_Et: [9., 22.]}
            """,
            "pion_trigger",
        ),
        # Gamma parameters
        (
            observable.extract_gamma_parameters,
            """
  eta: 1.0
  Et: [9., 22.]
  isolation:
    type: "neutral"
    R: 0.4
    Et_max: 0.1 # 10% of photon energy.
            """,
            "gamma_trigger",
        ),
        (
            observable.extract_z_parameters,
            """
  electron:
    pt: [20., null]
    eta: 2.47
  muon:
    pt: [20., null]
    eta: 2.5
  # Z candidates
  mass: [76., 106.]
  pt: [15., null]
            """,
            "z_trigger",
        ),
    ],
    ids=[
        "base parameters",
        "hadron parameters",
        "jet parameters",
        "pion parameters",
        "gamma parameters",
        "z boson parameters",
    ],
)
def test_extract_parameters(
    add_label: bool,
    inputs: tuple[callable[[dict[str, Any], str], observable.AllParameters], dict[str, Any]],
    caplog: Any,
) -> None:
    """Test extracting parameters from configurations.

    This implicitly tests `ParameterSpecs.from_config(...)`.

    These are supposed to be somewhat representative of real measurements, but they are not comprehensive.
    """
    f, config_str, label = inputs
    caplog.set_level(logging.INFO)

    y = ruamel.yaml.YAML()

    config = y.load(config_str)

    # Just run the function. Validation is a pain, so we just want to ensure it runs okay.
    res = f(config, label if add_label else "")  # noqa: F841

    # pytest.fail("Intentionally failing so that we can see the output")
