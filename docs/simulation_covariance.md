# Event-level covariance for Bayesian training

This opt-in sidecar reads already reconstructed, per-event scalar observable
lists. It does not reconstruct an observable, select a collision system, choose
experimental bins or change event weights. Sparse export uses SciPy, already
declared by this branch. Collection requires NumPy and PyArrow; it does not
import ROOT, FastJet or the Bayesian package.

The STAT writer retains `event_ID` and available vertex coordinates on both
filled and empty-event rows. Old parquet files without identities are rejected;
equal event counts or weights do not prove an event join. Do not regenerate
events just to add metadata, or invent IDs for an unverified legacy join.

## 1. Collect One Paired Cohort Chunk

```bash
python -m jetscape_analysis.analysis.simulation_covariance collect \
  --manifest collect.json --output moments.npz
```

For example, `collect.json` can contain synthetic/user-supplied inputs:

```json
{
  "cohort": "medium",
  "identity_columns": ["event_ID", "pt_hat", "vertex_x", "vertex_y"],
  "streams": [
    {"name": "point3", "file": "point3.parquet", "blocks": [
      {"name": "x", "columns": {"observable_x": 1}, "edges": [0, 1, 2]}
    ]},
    {"name": "point17", "file": "point17.parquet", "blocks": [
      {"name": "x", "columns": {"observable_x": 1}, "edges": [0, 1, 2]}
    ]}
  ]
}
```

All streams in a call represent the **same underlying generated events**,
paired across parameter points. The caller must establish that physical pairing
from generator provenance; coincidentally equal IDs are not sufficient evidence.
Rows must have identical declared identity fields, unique event IDs, and finite
nonnegative `event_weight_all`. Empty-event rows remain in the denominator.
Independent events/reference samples/centralities belong to separate calls.

Each stream has raw labels such as `point3:x:bin0` and `point3:events`.
Within an event, every bin, observable and paired stream enters **one** weighted
vector. The saved `outer` is the sum of vector outer products, not a per-jet
sum of squared weights and not a centered sample covariance. Thus multiple jets
in one event, paired parameter points and denominator fluctuations stay coupled.

`columns` may specify signed linear combinations, e.g. positive and subtraction
columns. All columns must exist; a null row means no entries. Only scalar lists
are accepted, and bins use `[low, high)`. Optional `normalization` counters are
`all_entries` or `underflow_and_visible` (values below the last edge, including
underflow). These create `point3:x:norm`. The default `events` uses the separate
all-event counter. No shape-normalization convention is inferred.

## 2. Normalize and Export

```bash
python -m jetscape_analysis.analysis.simulation_covariance export \
  --manifest export.json --output simulation_covariance.npz
```

The export manifest contains `design_ids`, `features`, `cohorts`, `spectra` and
`reference`. `features` is an ordered list of `{name, xmin, xmax}`. `cohorts`
maps independent cohort names to lists of collected chunk NPZ paths; chunks
within a cohort must have identical layouts. Repeated source-file hashes are
rejected across independent chunks/cohorts. Paths are relative to the manifest.

`spectra[design][bin]` and `reference[bin]` are explicit lists of ratio terms.
For example, after separately collecting a `reference` cohort with stream name
`pp` and the same histogram block, this complete `export.json` exports the first
bin for the two parameter points:

```json
{
  "design_ids": [3, 17],
  "features": [{"name": "observable_x", "xmin": 0, "xmax": 1}],
  "cohorts": {
    "medium": ["moments.npz"],
    "reference": ["reference_moments.npz"]
  },
  "spectra": [
    [[{"numerator": {"medium/point3:x:bin0": 1},
       "denominator": {"medium/point3:events": 1}, "scale": 2.5}]],
    [[{"numerator": {"medium/point17:x:bin0": 1},
       "denominator": {"medium/point17:events": 1}, "scale": 2.5}]]
  ],
  "reference": [
    [{"numerator": {"reference/pp:x:bin0": 1},
      "denominator": {"reference/pp:events": 1}, "scale": 3.0}]
  ]
}
```

Labels are prefixed by the cohort name. Coefficients define linear combinations
of raw statistics; `scale` is a fixed supplied cross section or other factor.
Multiple terms add **separately normalized** components; multiple labels in one
term pool **before** normalization. These operations are not interchangeable.
Reference terms use the same notation, normally with a separate shared sample.

If an observable is measured on only part of the event bank, partition events
into independent overlap and non-overlap cohorts. Collect all available columns
jointly in the overlap cohort. Map the full-statistics observable to both cohorts
and the subset observable only to its measured cohort, with its own denominator.
This retains the measured overlap covariance without treating unmeasured entries
as zero observations or scaling correlations by a sampling fraction.

Export differentiates the declared spectra and their reference ratios, then
propagates each independent cohort's full raw moments. A reference reused by
several designs creates cross-design covariance automatically; paired medium
events and cross-observable blocks are also retained. Sparse Jacobians avoid
expanding the normalization maps into dense raw-statistic matrices. The output
covariance itself remains dense.

The NPZ has `design_ids`, `observable_names`, `xmin`, `xmax`, `central_values`
and `covariance` with axes `(design, bin, design, bin)` in observable units.
This is the measured-noise input contract of JETSCAPE/Bayesian's
`simulation_noise.load_aligned_covariance`; select it with
`simulation_noise.covariance_file` and enable joint-group treatment there.
The identity-based loader can select/reorder designs and bins without discarding
their cross-covariance. Output files are never overwritten.

This is first-order event-Poisson ratio propagation, conditional on the supplied
normalization scales, cohort independence and fixed reconstruction/binning.
No cross-section uncertainty, theory discrepancy, inferred noise kernel,
resampling, clipping, jitter or covariance regularization is added. Sparse/empty
finite-MC bins still require scientific scrutiny; nonpositive reference bins and
materially non-PSD exported covariances fail rather than inventing errors.

Run `python -m pytest -q tests/test_simulation_covariance.py`. The optional
Bayesian handoff test runs when that package is installed; the rest is standalone.
