"""Hole subtraction with the negative recombiner.

Needs FastJet and heppy (fjcontrib, fjext): skipped without them. Synthetic particles only.
"""

from __future__ import annotations

import math

import pytest

fj = pytest.importorskip("fastjet")
pytest.importorskip("fjcontrib")
fjext = pytest.importorskip("fjext")

from jetscape_analysis.analysis import analyze_events_base_STAT  # noqa: E402


def particle(pt: float, phi: float, user_index: int):
    """Massless particle at eta = 0. As in the analysis, a negative user_index marks a hole."""
    p = fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi), 0.0, pt)
    p.set_user_index(user_index)
    return p


def find_jets(particles: list, R: float = 0.4) -> tuple:
    """Jet finding as in the negative_recombiner case of the analysis."""
    jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
    recombiner = fjext.NegativeEnergyRecombiner(-1)
    jet_def.set_recombiner(recombiner)
    cs = fj.ClusterSequence(particles, jet_def)
    # The cluster sequence and the recombiner have to stay alive with the jets
    return fj.sorted_by_pt(cs.inclusive_jets()), cs, recombiner


def signed_sum(particles: list) -> tuple[float, float, float]:
    """(px, py, E) of the particles minus the holes."""
    signs = [1 if p.user_index() > 0 else -1 for p in particles]
    return (
        sum(s * p.px() for s, p in zip(signs, particles, strict=True)),
        sum(s * p.py() for s, p in zip(signs, particles, strict=True)),
        sum(s * p.E() for s, p in zip(signs, particles, strict=True)),
    )


def test_net_negative_jet_has_negative_user_index():
    """The jet selection removes jets with a negative user_index: only those which are not jets must have one."""
    jet = [particle(100.0, 0.0, 1), particle(3.0, 0.1, -1)]
    two_holes = [particle(6.0, 2.0, -2), particle(4.0, 2.01, -3)]
    single_hole = [particle(7.0, 4.0, -4)]
    single_particle = [particle(5.0, 5.5, 2)]
    jets, _cs, _recombiner = find_jets(jet + two_holes + single_hole + single_particle)

    assert [j.pt() for j in jets] == pytest.approx([math.hypot(*signed_sum(jet)[:2]), 10.0, 7.0, 5.0], rel=1e-3)
    assert [j.user_index() >= 0 for j in jets] == [True, False, False, True]


def test_groomed_jet_subtracts_holes_merged_with_each_other():
    """C/A merges the two holes with each other first. The net-negative pair must still be subtracted."""
    prong_1 = [particle(60.0, 0.0, 1), particle(3.0, -0.10, -1), particle(2.0, -0.11, -2)]
    prong_2 = [particle(40.0, 0.2, 2)]
    jets, _cs, _recombiner = find_jets(prong_1 + prong_2)
    assert len(jets) == 1

    splitting, _, _gshop = analyze_events_base_STAT.calculate_groomed_jet(
        {"type": "soft_drop", "z_cut": 0.2, "beta": 0}, jets[0], 0.4, jet_collection_label="_negative_recombiner"
    )

    for prong, expected in [(splitting.harder(), prong_1), (splitting.softer(), prong_2)]:
        assert (prong.px(), prong.py(), prong.E()) == pytest.approx(signed_sum(expected))
    pair = splitting.pair()
    assert (pair.px(), pair.py(), pair.E()) == pytest.approx(signed_sum(prong_1 + prong_2))
