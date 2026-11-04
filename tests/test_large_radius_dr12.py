"""Focused reconstruction tests; require FastJet and the existing heppy fjext binding.

Load the actual methods without importing the unrelated ROOT plotting stack.
All particles and configurations below are synthetic, not production inputs.
"""
import ast
from collections import defaultdict
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

fj = pytest.importorskip('fastjet')
fjext = pytest.importorskip('fjext')
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'jetscape_analysis/analysis/analyze_events_STAT.py'


def load_method(path, name):
    method = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = dict(fj=fj, fjext=fjext, np=np)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


MEASURE = load_method(SOURCE, 'fill_large_radius_jet_observables')
FIND = load_method(SOURCE, 'find_jets_and_fill')


def particle(pt, phi=0., pid=211, sign=1):
    return pt, phi, pid, sign


def inputs(rows, label):
    positive, negative, pid_positive, pid_negative = [], [], [], []
    for pt, phi, pid, sign in rows:
        jet = fj.PseudoJet(pt * math.cos(phi), pt * math.sin(phi), 0., pt)
        particles, pids = (positive, pid_positive) if sign > 0 else (negative, pid_negative)
        pids.append(pid)
        jet.set_user_index(sign * len(pids))
        particles.append(jet)
    hadrons = positive + negative if label == '_negative_recombiner' else positive
    definition = fj.JetDefinition(fj.antikt_algorithm, .2)
    # Exercise the existing caller's default-marker inputs, not corrected test inputs.
    recombiner = fjext.NegativeEnergyRecombiner() if label == '_negative_recombiner' else None
    if recombiner is not None:
        definition.set_recombiner(recombiner)
    sequence = fj.ClusterSequence(hadrons, definition)
    return sequence, recombiner, hadrons, negative, np.array(pid_positive), np.array(pid_negative)


def context(names=('d12_atlas', 'dR12_atlas')):
    block = dict(jet_R=[.2], jet_R_reclustered=[1.], pt=[200., 251.], eta_cut=1.3, centrality=[[0, 10]])
    return SimpleNamespace(sqrts=5020, centrality_accepted=lambda ranges: True,
                           inclusive_jet_observables={name: dict(block) for name in names},
                           observable_dict_event=defaultdict(list))


def run(rows, label='', obj=None):
    obj = context() if obj is None else obj
    sequence, recombiner, hadrons, holes, pid_pos, pid_neg = inputs(rows, label)
    MEASURE(obj, fj.sorted_by_pt(sequence.inclusive_jets()), hadrons, holes, pid_pos, pid_neg, .2, label)
    return obj.observable_dict_event


def values(result, name='dR12_atlas', label=''):
    return result[f'inclusive_jet_{name}_R0.2{label}']


@pytest.mark.parametrize('label', ['', '_shower_recoil', '_negative_recombiner', '_constituent_subtraction'])
def test_positive_splitting_unchanged(label):
    result = run([particle(150), particle(70, .1)], label)
    np.testing.assert_allclose(values(result, label=label), [.1], atol=1e-12)
    np.testing.assert_allclose(values(result, 'd12_atlas', label), [7.], atol=1e-12)


def test_negative_track_subtracts_from_prong_momentum():
    result = run([particle(150), particle(100, .1), particle(30, .18, sign=-1)], '_negative_recombiner')
    px, py = 100*math.cos(.1)-30*math.cos(.18), 100*math.sin(.1)-30*math.sin(.18)
    # Signed aggregates can be spacelike: use FastJet's rapidity distance,
    # not just the azimuth difference, for this independent four-vector oracle.
    prong = fj.PseudoJet(px, py, 0., 70.)
    dr = fj.PseudoJet(150., 0., 0., 150.).delta_R(prong)
    np.testing.assert_allclose(values(result, label='_negative_recombiner'), [dr], atol=1e-12)
    np.testing.assert_allclose(values(result, 'd12_atlas', '_negative_recombiner'),
                               [math.hypot(px, py)*dr], atol=1e-12)


def test_negative_neutral_pid_not_looked_up_in_positive_array():
    result = run([particle(150), particle(100, .1), particle(30, .18, pid=22, sign=-1)], '_negative_recombiner')
    np.testing.assert_allclose(values(result, label='_negative_recombiner'), [.1], atol=1e-12)


def test_negative_only_cluster_is_not_a_physical_jet():
    result = run([particle(150, sign=-1), particle(70, .1, sign=-1)], '_negative_recombiner')
    assert not values(result, label='_negative_recombiner')


def test_merged_negative_remnant_does_not_become_a_calorimetric_seed():
    result = run([particle(150), particle(70, .1), particle(80, .5, pid=22, sign=-1),
                  particle(80, .501, pid=22, sign=-1)], '_negative_recombiner')
    np.testing.assert_allclose(values(result, label='_negative_recombiner'), [.1], atol=1e-12)


def test_merged_negative_tracks_keep_their_sign():
    result = run([particle(150), particle(100, .1), particle(15, .18, sign=-1),
                  particle(15, .1801, sign=-1)], '_negative_recombiner')
    px = 100*math.cos(.1) - 15*math.cos(.18) - 15*math.cos(.1801)
    py = 100*math.sin(.1) - 15*math.sin(.18) - 15*math.sin(.1801)
    dr = fj.PseudoJet(150., 0., 0., 150.).delta_R(fj.PseudoJet(px, py, 0., 70.))
    np.testing.assert_allclose(values(result, label='_negative_recombiner'), [dr], atol=1e-12)


def test_shower_recoil_shared_hole_is_subtracted_once():
    result = run([particle(140), particle(100, .3), particle(20, .15, pid=22, sign=-1)], '_shower_recoil')
    np.testing.assert_allclose(values(result, label='_shower_recoil'), [.3], atol=1e-12)


def test_negative_prong_cannot_pass_soft_drop():
    result = run([particle(280), particle(70, .8, sign=-1)], '_negative_recombiner')
    assert not values(result, label='_negative_recombiner')


def test_signed_charged_fraction_cut():
    result = run([particle(250, pid=22), particle(50), particle(40, .1),
                  particle(100, .05, sign=-1)], '_negative_recombiner')
    assert not values(result, label='_negative_recombiner')


@pytest.mark.parametrize('pid', [12, 14, 16])
def test_noncalorimetric_particles_do_not_change_jet_pt_selection(pid):
    result = run([particle(150), particle(70, .1), particle(1000, .05, pid=pid)])
    assert len(values(result)) == 1


def test_muons_are_tracks_but_not_calorimeter_inputs():
    result = run([particle(150, pid=22), particle(70, .1, pid=22),
                  particle(60, .04, pid=13), particle(50, .13, pid=-13)])
    np.testing.assert_allclose(values(result), [.09], atol=1e-12)


def test_association_ghosts_do_not_move_upper_selection_boundary():
    result = run([particle(150), particle(101-1e-10)])
    assert len(values(result)) == 1
    assert not values(run([particle(150), particle(101)]))


def test_optional_blocks_and_individual_output():
    assert not run([particle(150), particle(70, .1)], obj=context(()))
    obj = context(('dR12_atlas',))
    result = run([particle(150), particle(70, .1)], obj=obj)
    assert set(result) == {'inclusive_jet_dR12_atlas_R0.2'}
    obj.inclusive_jet_observables['dR12_atlas']['enabled'] = False
    obj.observable_dict_event.clear()
    assert not run([particle(150), particle(70, .1)], obj=obj)


def test_inconsistent_joint_selections_fail():
    obj = context()
    obj.inclusive_jet_observables['dR12_atlas']['pt'] = [200., 300.]
    with pytest.raises(ValueError, match='identical'):
        run([particle(150), particle(70, .1)], obj=obj)


def test_event_level_fill_precedes_selector_and_is_full_jet_only():
    seq, recombiner, hadrons, holes, pid_pos, pid_neg = inputs([particle(200), particle(40, .5)], '')
    obj = context()
    measured = []
    obj.fill_large_radius_jet_observables = lambda *args: (measured.append(True), MEASURE(obj, *args))
    obj.analyze_inclusive_jet = lambda *args, **kwargs: None
    obj.dijet_observables = obj.semi_inclusive_chjet_observables = {}
    definition = fj.JetDefinition(fj.antikt_algorithm, .2)
    selector = fj.SelectorPtMin(100.)
    FIND(obj, hadrons, holes, pid_pos, pid_neg, definition, selector, .2, '', full_jet=True)
    assert len(values(obj.observable_dict_event)) == 1  # The 40 GeV small jet is needed.
    assert len(measured) == 1
    FIND(obj, hadrons, holes, pid_pos, pid_neg, definition, selector, .2, '', full_jet=False)
    assert len(measured) == 1


def test_parquet_scalar_storage_uses_existing_histogram_route(tmp_path):
    import pandas as pd
    result = run([particle(150), particle(70, .1)])
    name = 'inclusive_jet_dR12_atlas_R0.2'
    path = tmp_path / 'observables.parquet'
    pd.DataFrame([result]).to_parquet(path)
    calls = []
    obj = SimpleNamespace(observables_df=pd.read_parquet(path), is_AA=False,
                          histogram_1d_observable=lambda *a, **k: calls.append(k))
    histogram = load_method(ROOT / 'plot/histogram_results_STAT.py', 'histogram_observable')
    histogram(obj, column_name=name, bins=np.array([0., .2, 1.]), centrality=[0, 10])
    assert len(calls) == 1 and calls[0]['column_name'] == name
