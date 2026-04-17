from __future__ import annotations

import logging
from typing import Any

import pytest
from jetscape_analysis.data_curation import observable

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "spec",
    [
        observable.CentralitySpec(0.0, 10.0),
        observable.PtSpec(10.0, 20.0),
        observable.EtaSpec(1.0),
        observable.JetRSpec(0.4),
        observable.SoftDropSpec(0.2, 0.1),
        observable.DynamicalGroomingSpec(1.0),
        observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.1)),
        observable.JetAxisDifferenceSpec("WTA-SD", grooming_settings={"z_cut": 0.2, "beta": 0.5}),
        observable.AngularitySpec(2.0),
        observable.SubjetRSpec(0.1),
    ],
    ids=lambda x: f"{x!r}",
)
def test_param_spec_round_trip(spec: observable.ParameterSpec, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    logger.info(f"{spec.encode()=}")
    assert spec.decode(spec.encode())


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
        # Eta_R
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0)]),
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)]),
        observable.EtaRSpecs(values=[observable.EtaSpec(1.0), observable.EtaSpec(1.2)], label="jet"),
        # Jet R
        observable.JetRSpecs(values=[observable.JetRSpec(0.2)]),
        observable.JetRSpecs(values=[observable.JetRSpec(0.2), observable.JetRSpec(0.4), observable.JetRSpec(0.6)]),
        observable.JetRSpecs(values=[observable.JetRSpec(0.2), observable.JetRSpec(0.4)], label="jet"),
        # SoftDrop
        observable.SoftDropSpecs(values=[observable.SoftDropSpec(0.2, 0.1)]),
        observable.SoftDropSpecs(values=[observable.SoftDropSpec(0.2, 0.1), observable.SoftDropSpec(0.1, 0.0)]),
        observable.SoftDropSpecs(
            values=[observable.SoftDropSpec(0.2, 0.1), observable.SoftDropSpec(0.1, 0.0)], label="jet"
        ),
        # DyG
        observable.DynamicalGroomingSpecs(values=[observable.DynamicalGroomingSpec(1.0)]),
        observable.DynamicalGroomingSpecs(
            values=[observable.DynamicalGroomingSpec(1.0), observable.DynamicalGroomingSpec(2.0)]
        ),
        observable.DynamicalGroomingSpecs(
            values=[observable.DynamicalGroomingSpec(1.0), observable.DynamicalGroomingSpec(2.0)], label="jet"
        ),
        # Jet-axis difference
        observable.JetAxisDifferenceSpecs(
            values=[observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.1))]
        ),
        observable.JetAxisDifferenceSpecs(
            values=[
                observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.1)),
                observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.0)),
            ]
        ),
        observable.JetAxisDifferenceSpecs(
            values=[
                observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.1)),
                observable.JetAxisDifferenceSpec("WTA-SD", observable.SoftDropSpec(0.2, 0.0)),
            ],
            label="jet",
        ),
        # Angularities
        observable.AngularitySpecs(values=[observable.AngularitySpec(1.0)]),
        observable.AngularitySpecs(values=[observable.AngularitySpec(1.0), observable.AngularitySpec(2.0)]),
        observable.AngularitySpecs(
            values=[observable.AngularitySpec(1.0), observable.AngularitySpec(2.0)], label="jet"
        ),
        # Subjet R
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0)]),
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0), observable.SubjetRSpec(2.0)]),
        observable.SubjetRSpecs(values=[observable.SubjetRSpec(1.0), observable.SubjetRSpec(2.0)], label="jet"),
    ],
    ids=lambda x: f"{x!r}",
)
def test_param_specs_round_trip(specs: observable.ParameterSpecs, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    logger.info(f"{specs.encode()=}")
    assert specs.decode(specs.encode())
