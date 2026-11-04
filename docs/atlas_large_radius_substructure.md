# ATLAS large-radius charged-track splitting observables

This ports Dhanush Hangal's `fill_large_radius_jet_observables` from
[PR #21](https://github.com/JETSCAPE/JETSCAPE-analysis/pull/21), commit
`3d9d3ba9ad014a8876cfc716919cec87514d2961`, to the Hybrid branch's existing
configuration and output names, including Jingyu Zhang's event-level
reconstruction and storage corrections. The measurement is arXiv:2504.04805, sections 4-5.
It does not add experimental data, Bayesian configurations or prediction tables.

The upstream reconstruction is retained: anti-kt R=0.2 jets above 35 GeV and
within abs(eta)<3 seed R=1 jets; charged tracks above 4 GeV and within
abs(eta)<2.5 are ghost-associated; their scalar pT fraction must exceed 0.2.
The tracks are reclustered with kt R=2.5, then manually declustered along the
harder branch with strict z>0.15. There is no C/A reclustering. The outputs are
the accepted splitting's dR12 and sqrt(d12)=min(prong pT)*dR12.

## Hybrid subtraction

For `_negative_recombiner`, both calorimetric and charged-track clustering use
the existing `fjext.NegativeEnergyRecombiner(-1)`. The explicit negative marker
is necessary: its default marker is zero, which does not preserve a negative
merged cluster's sign in subsequent recombinations. This correction is local
to these new observables; the other existing observables are unchanged.

Charged negative wake above the track cut enters with its negative PID index
and subtracts from the charged fraction and recombined prong four-vectors.
Net-negative small jets and net-negative hardest charged roots cannot be
physical selected jets. A negative branch cannot be tagged as a resolved prong;
declustering continues into its positive sibling. Otherwise the upstream
hardest-root and harder-branch rules are unchanged. This is the model's signed
subtraction convention, not an experimental prescription for negative particles.

The upstream `_shower_recoil` alternative retains positive tracks and its
pT-only hole correction, counting a shared calorimetric hole once in the
large jet. `_constituent_subtraction` uses the already-subtracted particles.
Muons and neutrinos are excluded from calorimetric inputs locally; charged
muons remain eligible tracks. Association ghosts do not enter the reported jet
four-vector or its selection cuts.

## Opt-in Configuration

Append either or both blocks to the existing `inclusive_jet` section. Do not
replace the other blocks. Both are absent, and therefore inactive, by default.
The global `jet_R` list must include 0.2. If both observables are enabled they
must use identical jet selections. Only one jet-pT interval is supported.

```yaml
d12_atlas: &atlas_large_R
  enabled: true
  jet_R: [0.2]
  jet_R_reclustered: [1.0]
  pt: [200., 251.]
  eta_cut: 1.3
  centrality: [[0, 10]]
dR12_atlas:
  <<: *atlas_large_R
```

These are event-analysis blocks, not complete experimental plotting inputs.
Supply the measurement's histogram binning and plotting metadata separately;
no private ATLAS release is distributed here. The scalar parquet columns are
`inclusive_jet_d12_atlas_R0.2{collection}` and
`inclusive_jet_dR12_atlas_R0.2{collection}`, using the existing one-dimensional
histogram route. No change to event weights or cross-section normalization is
introduced. Raw analysis columns do not themselves constitute normalized RAA.

## Tests

With the normal FastJet/heppy bindings available:

```bash
python -m pytest -q tests/test_large_radius_dr12.py
```

Tests cover positive-event equivalence, signed four-vector subtraction, charged
fraction, negative-only jets/prongs, independent positive/negative PID arrays,
muon/neutrino handling, ghost-boundary effects, event-level routing before the
small-jet selector, optional configuration and scalar parquet/histogram routing.
