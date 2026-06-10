# Observable-encoding migration — work in progress

Raymond started an in-flight migration on `dev-aggregation`. This file
documents the state as of 2026-05-19, what was fixed to keep the pipeline
running on the new YAML, and the work that remains to complete the
end-to-end conversion.

The user-facing read of this is: **the analyzer now completes; the
histogrammer runs but silently produces no histograms for most observables
because the binning lookup hasn't been migrated yet.** Plan for finishing
the migration is at the end.

See also **`OBSERVABLE_EDGE_CASES.md`** for the registry of HEPData/observable
edge cases (missing pp, only-spectra, no-data-block, center-value tables, …) and
exactly how each is handled (fallback / protect / skip-null / pending).

## Raymond's design (one paragraph)

The `data_curation/observable.py` module is meant to be the single source
of truth for parameter encoding. Each observable's YAML block declares
its parameters (R, eta, pt bins, grooming settings, …) plus a `data:`
sub-block that maps HEPData tables to parameter combinations. The
analysis, histogramming, and plotting layers are supposed to ask the
`Observable` object — via `obs.encode_name_for_storing_in_file(...)` and
`obs.essential_parameters()` — for histogram names and parameter
combinations, instead of hand-building f-strings. "Essential parameters"
are those that vary (`len(values) > 1`); they uniquely identify a table.

## Pipeline migration state

| Layer | YAML schema | Histogram naming | Binning lookup |
|---|---|---|---|
| YAML config | new (`data:` blocks, `jet.R`, `jet.grooming_settings`) | n/a | n/a |
| Analyzer (`analyze_events_STAT.py`) | reads new schema ✓ | groomed → new `obs.encode_name_for_storing_in_file`; rest → old f-strings | n/a |
| Histogrammer (`plot/histogram_results_STAT.py`) | reads new schema after rename ✓ | old f-strings everywhere | **broken — `bins_from_config` still looks for `block["bins"]` / `block["hepdata"]`** |
| Plotter (`plot/plot_results_STAT.py`) | reads new schema after rename ✓ | old f-strings | broken — `plot_results_STAT_utils` HEPData lookup uses old `hepdata_*_dir` keys |

## What was fixed in this commit

1. **Analyzer crash fixes** (`analyze_events_STAT.py`):
   - Removed stray `[0]` in `pt_ch_cms` pt unpack.
   - Renamed `["soft_drop"]` → `["jet"]["grooming_settings"]` at all groomed observable sites (`ktg_alice`, `zg_alice`, `angularity_alice`, `mass_alice`, `mg_cms`, `zg_cms`, `rg_atlas`, `rg_cms`). Upstream commit `5a50a6b` renamed the YAML key but missed the analyzer.
   - Fixed `angularity_alice` swapped indexing: `["pt"]["jet"]` → `["jet"]["pt"]` and `["R"]` → `["jet"]["R"]`.
   - For multi-pt-bin groomed observables (`mg_cms`, `zg_cms`, `rg_atlas`, `angularity_alice`, `mass_alice`): added `jet_pt=PtSpec(low, high)` to the `obs.encode_name_for_storing_in_file` kwargs, because their data spec declares `jet_pt` as essential (multiple bins). Without this, the encoder raised `KeyError: 'jet_pt'`. Each per-event fill now finds the bin the jet falls into and passes a single `PtSpec(bin_low, bin_high)`. Data shape changed from `[jet_pt, value]` (2D into one histogram) to `value` (1D into per-bin histograms).
   - Single-pt-bin groomed observables (`ktg_alice`, `zg_alice`, `tg_alice`, `axis_alice`) were left untouched. They don't crash because their `jet_pt` is not essential (only one bin).

2. **Histogrammer schema renames** (`plot/histogram_results_STAT.py`):
   - All `block["jet_R"]` → `block["jet"]["R"]`.
   - All `block["pt"]` in jet contexts → `block["jet"]["pt"]`.
   - `block["soft_drop"]` → `block["jet"]["grooming_settings"]`.
   - `block["axis"]`, `block["kappa"]` → `block["jet"]["axis"]`, `block["jet"]["kappa"]`.
   - Removed the partial flatten-shim in `histogram_hadron_trigger_chjet_observables` (it copied `block["jet"]["R"]` to `block["R"]`, but the code read `block["jet_R"]` — so the shim never helped).

3. **Plotter schema renames** (`plot/plot_results_STAT.py`):
   - Same shape as histogrammer renames in `plot_jet_observables` and `plot_hadron_trigger_chjet_observables`.
   - `init_observable` now defensively checks `block["jet"]["pt"]` and `block["hadron"]["pt"]` for the pt list.

## What's still broken / deferred

### 1. Histogrammer binning lookup (`plot_results_STAT_utils.PlotUtils.bins_from_config`)

Old logic:
- if `"bins"` in block → use it directly
- elif `"hepdata"` in block → read from the HEPData **ROOT** file referenced

In the new YAML neither key is present on most observables (3 still have explicit `bins:`, 3 still have a top-level `hepdata:` filename). The other ~49 observables have only the new `data:` block, which references HEPData by `inspire_hep_id` + `table` + `index`. The histogrammer doesn't yet know how to consume that.

**Important: the new HEPData source is YAML, not ROOT.** The
`hard-sector-data-curation` submodule (sibling clone at
`/afs/cern.ch/work/z/zhangj/jetscape-analysis/hard-sector-data-curation/`
right now since the submodule isn't initialized) stores HEPData in
per-table `.yaml` files:

```
hard-sector-data-curation/
├── hepdata_database.yaml                          # observable → directory
└── data/5020/inclusive_jet/mg_cms/
    └── HEPData-ins1672962-v1-yaml/
        ├── Table1.yaml
        ├── Table2.yaml
        └── …
```

So the binning lookup migration is actually two changes at once:
- read a different YAML key (`data:` block instead of `hepdata:` filename),
- read a different *file format* (HEPData YAML instead of HEPData ROOT).

**Possible paths to fix:**
- Quick: hand-add explicit `bins:` to every observable block (tedious, defeats Raymond's design).
- Right: extend `bins_from_config` to read from the new `data:` block via the `data_curation` module, which already parses HEPData v2 YAML and exposes binning. See `jetscape_analysis/data_curation/data.py:parse_binning_block` (~line 469) and `jetscape_analysis/data_curation/hepdata_utils.py`. The same helper should replace `bins_from_hepdata` in `plot_results_STAT_utils.py`.

Without this fix, after running through the analyzer + histogrammer, the resulting ROOT file will contain zero (or near-zero) histograms despite the analyzer doing all the work correctly.

**Observables still on the ROOT-file path** (kept working in this commit batch):
- `xj_gamma_atlas`, `xj_gamma_cms` (top-level `hepdata:` filename + `pt_gamma_bins:` + `hepdata_{pp,AA}_hname:` still present).
- 1 other (search `^    hepdata: ` in `config/STAT_5020.yaml` for the full list).

Everything else expects YAML.

### 2. Groomed observables in histogrammer + plotter

The analyzer now writes new-encoder names for groomed observables, e.g.
`inclusive_jet_mg_cms_jet_pt_140.0_160.0_jet_R_0.4_jet_grooming_settings_SD_z_cut_010_beta_0_shower_recoil`.
The histogrammer still hand-builds the old-style names
(`inclusive_jet_mg_cms_R0.4_zcut0.1_beta0_shower_recoil`) when looking
for keys in the analyzer's output — so groomed observables are dropped
on the floor.

This is the bigger structural piece Raymond wanted: rewrite the
histogrammer and plotter to walk `obs.essential_parameters()` and use
`obs.encode_name_for_storing_in_file(...)` instead of nested for-loops.

### 3. Plotter HEPData lookup (`plot_results_STAT_utils`)

`bins_from_hepdata` and the comparison-plot HEPData functions all read
`block["hepdata_pp_dir"]`, `block["hepdata_AA_dir"]`,
`block["hepdata_pp_gname"]`, etc. These are commented out everywhere in
the new YAML; the new equivalent lives in `data.pp.spectra.tables[i].table`
and friends. Same fix shape as the histogrammer binning lookup —
delegate to `data_curation`.

### Targeted patches landed on top of the schema-rename commit

- **`xj_gamma_atlas` and `xj_gamma_cms`** were re-enabled and patched
  end-to-end so a student can run them on the new YAML. Fixed:
  - YAML: `enabled: true`; `dPhi: 7./8.` → `dPhi: 0.875` (was a YAML
    string, broke the `* np.pi` arithmetic).
  - Analyzer: full schema rewrite for these two observables — old keys
    (`gamma_pT`, `gamma_eta`, `isolation_R`, `R`, `jet_eta`, `jet_pT`,
    `jet_deltaphi`) → new nested locations (`trigger.pt`, `trigger.eta`,
    `trigger.isolation.{R,type,Et_max_pp,Et_max_AA}`, `jet.R`,
    `jet.eta`, `jet.pt`, `dPhi`). Also adjusted shape expectations: new
    `trigger.pt` is a list of bin edges (used min/max for the trigger
    cut), `trigger.eta` and `jet.eta` are scalars (used as upper bound
    with min=0), `jet.R` is a list (took `[0]`).
  - Analyzer: `.append(photon.Et(), xj)` (Python TypeError) →
    `.append([photon.Et(), xj])`. Four sites.
  - Histogrammer: `photon_jet_xj_{atlas,cms}_...` → `gamma_trigger_jet_xj_{atlas,cms}_...`
    column-name reconciliation (analyzer always wrote
    `gamma_trigger_jet_*` — the histogrammer was looking for a stale
    name that never existed in the dict).
  - Histogrammer: the `column_names` list at line ~423 had no `f`
    prefix, so `{jet_R}` was being looked up as a literal string. Fixed.

  These two observables now have explicit `hepdata:` filename +
  `pt_gamma_bins:` + `hepdata_{pp,AA}_hname:` in the YAML, so the
  histogrammer's binning lookup works for them even though it's broken
  for most other observables.

  **Pre-flight check before running**: the student needs the HEPData
  ROOT file at `data/STAT/5020/gamma_trigger_jet/{xj_gamma_atlas,xj_gamma_cms}/HEPData-ins*.root`.
  Download from the inspire_hep link in the YAML if missing.

### 4. Per-observable cleanup items (Raymond's note)

- **Split `mass_alice` (5020) into `mass_alice` + `mg_alice`** — currently
  it mixes ungroomed (`z_cut=0, beta=0` sentinel) and groomed under one
  block. The 2.76 TeV `mass_alice` (ungroomed) stays untouched. Similarly
  for angularity → `angularity_groomed_alice`.
- **Spectra / ratio / double_ratio types** — only `spectra` and `ratio`
  are currently recognized. Jet RAA observables want `double_ratio` as a
  third artifact type.
- **Bin-edges source of truth** — once a binning lookup is in place,
  decide if the canonical bin edges live in `data:.spectra.x_axis` or
  in the old top-level `bins:` field. Pick one.

## Step 1 — DONE (2026-06-01): `bins_from_config` now reads the `data:` block

Implemented `PlotUtils.bins_from_data_block` in `plot/plot_results_STAT_utils.py`
+ a `if "data" in block:` branch in `bins_from_config`. Verified end-to-end on the
PbPb 5020 small sample: 47/47 targeted non-groomed histograms now populate
(hadron RAA `pt_ch_alice`/`pt_pi_alice`/`pt_ch_cms`, inclusive-jet RAA
`pt_atlas`/`pt_cms` all R, `Dz_atlas`) — previously ~zero.

**Corrections to this doc found while implementing:**
- **Bin edges are NOT in `data.spectra.x_axis`.** That field only holds
  `{label, range, log}` for display. The real edges live in the HEPData per-table
  YAML at `independent_variables[0].values` (`{low, high}` per bin). The new code
  reads them there, reusing the same pattern as `build_tables.py:write_data_table`.
- **`parse_binning_block` (data.py:469) is a stub** and only ever applied to a
  literal `bins:` sub-block — NOT the HEPData path. Bypassed it.
- **`BASE_DATA_DIR` selection was broken** for an uninitialized submodule: the
  empty placeholder dir `data/hard-sector-data-curation/` passed `.is_dir()` and
  shadowed the populated sibling clone. Changed to require the database file to
  exist. (`hepdata_utils.py`)

**Environment gotchas (analysis container `stat_local_gcc_v5.2.sif`):**
- Container is missing `ruamel.yaml` and `requests`; made both lazy imports in
  `hepdata_utils.py` so the histogrammer can import `data_curation`. The new
  binning code parses the DB with pyyaml directly (not `read_database`, which
  needs ruamel). Analyzer additionally needs `scipy` (pip `--target` to /tmp).
- Run the analyzer with the heppy env (`/jetscapeOpt/heppy/modules/heppy/1.0`):
  set PYTHONPATH/LD_LIBRARY_PATH to pythia8309 + root + hepmc2 + lhapdf6 + fastjet
  3.3.4 (lib + `local/lib/python3.10/dist-packages`) + heppy cpptools. The
  histogrammer needs only root on the path (no fastjet).

**Newly surfaced, deferred to Step 4 (substructure):** enabling the `data:`-block
binning made `inclusive_jet/axis_cms` (a WTA-axis substructure observable) book
histograms; its fill path appends ~165 null ROOT objects to `output_list`.
`write_output_objects` now skips nulls (so one bad observable can't corrupt the
whole file), but the root cause in the axis/substructure fill path is unfixed —
address with the groomed/substructure rewrite in Step 4.

**pp analyzer note:** the pp small-sample analyzer segfaults at event 0 in the
fastjet/C++ layer (exit 139) — independent of this change (PbPb runs clean on the
same code/env). Needs separate investigation before the pp arm can be tested.

## Step 4 — DONE (2026-06-02): encoder-name migration, histogrammer + plotter

Groomed/substructure jet observables (`mg_cms, zg_cms, rg_atlas, axis_cms,
ktg_alice, zg_alice, tg_alice, axis_alice`) are now on **one name path
end-to-end**: the analyzer, histogrammer (Step 4 ③, `5a4f30f`), and plotter
(Step 4 ④, `47583d2`) all derive the histogram name from
`obs.encode_name_for_storing_in_file(...)` instead of hand-built f-strings.

- The migrated set lives in `plot_results_STAT_utils.ENCODER_MIGRATED_JET_OBSERVABLES`
  (shared by histogrammer + plotter — single source of truth).
- Per-grooming / per-axis HEPData table selection (`data_block_params`) is wired
  into both the binning path and the overlay path (`tgraph_from_data_block`).
- `_encoder_column_name` is duplicated in the histogrammer and the plotter
  (lockstep copies); consolidate into a shared helper in a future refactor.
- Verified e2e (small samples): plotter-built names resolve exactly against the
  histogrammer output (AA mg_cms 6 / zg_cms 3 / axis_cms 12; pp mg_cms 24 /
  zg_cms 3 / ktg·zg·tg_alice 2 / axis_alice 6). Non-migrated unaffected.

**`mass_alice` / `angularity_alice`** are intentionally OFF the migrated set
(ungroomed↔groomed split = Step 5). **`axis_alice`** keeps its legacy ROOT
overlay path (legacy `hepdata_*` keys → strip in Step 6). See
`OBSERVABLE_EDGE_CASES.md` C5 / A9 / C6 / C7.

### Full render now COMPLETES (2026-06-02): pre-existing render-path fixes E1/E2/E3
### ✅ RESOLVED (2026-06-02 PM): AA R_AA bin-mismatch bug fixed — see "AA R_AA binning fix" below

**RESOLVED.** The suspect AA R_AA was hypothesis (b): the histogrammer binned the AA arm
on the HEPData `ratio` table but the pp arm on the `spectra` table → different, non-nested
edges → `plot_RAA`'s `AA.Divide(h_pp)` failed with "Cannot divide histograms with different
number of bins" (16×, AA stage only) → R_AA wrong/empty. Fixed by booking a pp
R_AA-denominator on the AA `ratio` binning (`_raa_denom`); see the dedicated section below.
(The other hypotheses were not the cause: (a) eta_cut is applied to both arms and cancels in
the ratio; (c) limited-centrality sample is real but secondary; (d) display-only.)

The first complete render after Step 4 ④ exercised the AA→R_AA path for the first
time and hit three **pre-existing** render-layer bugs (separate from the encoder
migration), all now fixed — see `OBSERVABLE_EDGE_CASES.md` section E:
- **E1** (`f13a717`): acceptance-cut schema rename — `init_common_settings` read
  the legacy top-level `eta_cut`/`eta_cut_R`/`y_cut`; the new YAML nests them as
  `hadron.eta` / `jet.eta` / `jet.eta_R` / `jet.rapidity`. η (pseudorapidity) and
  y (rapidity) kept separate. Now reads the nested keys (legacy as fallback).
- **E2** (`51e1ffa`): AA branch never defaulted `self.ytitle` / `self.y_ratio_min/max`
  → `plot_RAA` AttributeError. Defaulted, mirroring the pp branch.
- **E3** (`51e1ffa`): `write_experimental_data` hard-raised on a data/hist binning
  mismatch (`pt_cms`), aborting the run after the R_AA PDF was already saved. Now
  warn-skips the ancillary `Data_*.dat` table.

Verified: real plotter renders pp (128 PDFs) + AA→R_AA (25 PDFs) clean, exit 0.
**`axis_alice` is excluded** from a whole-config render — it still has legacy
`hepdata_*` keys pointing at HEPData ROOT files not on disk (the only enabled
observable in that state); resolve in Step 6 by stripping those keys so it uses
its `data:` block.

## AA R_AA binning fix — DONE (2026-06-02 PM)

**Bug:** the AA arm was binned on the HEPData `ratio` table but the pp arm on the `spectra`
table → different, NON-nested edges → `plot_RAA`'s `AA.Divide(h_pp)` failed (16× "Cannot
divide histograms with different number of bins", AA stage only), so R_AA was wrong/empty for
`pt_ch_cms`, `pt_ch_atlas`, `pt_cms` R0.8/R1.0, `Dz/Dpt_atlas`, etc. (`pt_ch_alice`/`pt_pi_alice`
happened to have ratio==spectra in the test sample, so they divided fine and masked the bug.)

**Fix (re-fill, NOT resample):** the histogrammer now books a SECOND pp histogram on the AA
`ratio` binning by re-filling the same observable column, named with a `_raa_denom` suffix
(`HistogramResults.maybe_book_raa_denom`, pp-run only; skipped when ratio==spectra, no ratio
table, or non-monotonic edges e.g. `pt_y_atlas`). The plotter scales it through the identical
chain (`_scale_one_histogram`), persists `jetscape_distribution_raa_denom_{label}` into pp
`final_results.root`, and `plot_RAA` divides by it — falling back to the spectra histogram when
no denom exists (preserves prior behavior for ratio==spectra). `_raa_denom` is excluded from
`keys_to_plot` and `write_experimental_data`. Files: `plot/histogram_results_STAT.py`,
`plot/plot_results_STAT.py`.

**Also (TASK B):** `_save_logxy_twin` emits a log-x/log-y `_logxy.pdf` twin for distribution/
spectra plots only (NOT R_AA ratios — degenerate log-y on a 0–2 range).

**Verified e2e (small sample):** analyzer/histogrammer/plotter all exit 0, **0** "Cannot divide"
errors (was 16), pp 255 PDFs (128 + 127 `_logxy`), AA 25. Multi-agent adversarial review +
runtime checks passed: pp `_raa_denom` edges bit-identical to AA (incl. the genuine mismatch
cases pt_cms R0.8/R1.0 5-vs-4, pt_atlas R0.4 15-vs-14), 0 `_raa_denom` in the AA arm, all AA
Divides finite (means ~0.69–0.95), no silent denom misses. One confirmed LOW-severity latent
defect (Dz `_raa_denom` re-booked the binning-independent `_Njets` companion under a duplicate
name — dormant in 5020, would activate for 2760 Dz blocks) fixed with an `and not name_suffix`
guard. Residual (pre-existing, not introduced): pp `_raa_denom` is persisted under the bare key
(no collection_label) — correct only because the pp run always uses `collection_label=""`.

Note: AA=25 (vs pp 128) is the limited-centrality single-chunk sample, not a bug — the AA
plotter skips ~93 (observable, centrality) combos whose AA MC histogram is empty for this chunk
(only [10,20]/[10,30] populate). Run more PbPb chunks across centralities for fuller coverage.

## Plotter cosmetics + data-overlay fixes — DONE (2026-06-02 PM)

All in `plot/plot_results_STAT.py` + `plot/plot_results_STAT_utils.py` (+ a config flag). Verified
e2e (exits 0, 0 divide errors, 0 invalid-index); reviewed via `/code-review high` (2 CONFIRMED +
6 PLAUSIBLE findings, all addressed or documented as intentional).

- **Content-driven y-auto-range** (`_content_extrema` + `_auto_y_range`): the upper distribution panel
  is ranged from the actual drawn content (hists + data, errors included); log vs linear auto-chosen by
  dynamic range. Replaces the old `[0, 1.99]` linear default that left many curves (e.g. `pt_y_atlas`,
  values ~1e-3) invisible. Explicit config y-ranges (`y_min_pp` etc.) still win.
- **Span-adaptive log headroom** (`_log_yrange`): top factor ×1e4 for steep pT spectra (≥4 decades),
  ×50 otherwise, so the highest points clear the title/legend.
- **R_AA fixed y-range 0–3** (`plot_RAA`) so the legend clears the points.
- **log-x/log-y twin** (`_save_logxy_twin`): emits `<name>_logxy.pdf` for distribution plots only (NOT
  R_AA); log-x only when the x-range is > 0 (so `m_g`/`z_g` get a non-empty log-y twin instead of an
  empty log-x plot); restores pad log-state + a log-appropriate y-range for the twin.
- **Display-only area normalization** (`area_normalize: true` config flag, currently on
  `hadron/pt_ch_atlas`): when the data is an absolute cross-section (mb GeV⁻²) but the MC is a per-event
  yield, scale a DISPLAY CLONE of the MC so its integral matches the data over the common range where
  BOTH have content (i.e. above the MC pT cut). The original MC is persisted to `final_results.root`
  (R_AA stays absolute); the MC/data ratio is recomputed from the clone; the plot is annotated
  "MC area-normalized to data". Warns if requested but no overlapping range / no data graph.
- **Robust MC/data ratio** (`divide_histogram_by_tgraph` rewrite): iterates the DATA graph points and
  matches each to its hist bin via `h.FindBin` (point placed at the bin center), instead of the old
  index-aligned `truncate_tgraph` that bailed the whole ratio to `None` on ANY mismatch. Fixes
  observables whose measured spectrum leaves bins unpublished (`'-'`) so the data graph has fewer points
  than the MC (e.g. `inclusive_jet/pt_cms` R0.8/R0.2). Intentional behavior changes vs the old path:
  omits empty-MC bins (no longer 0-valued points), tolerates count mismatches. `truncate_tgraph` (the
  `is_AA` overlay raise-guard) is untouched.
- **ktg_alice data-overlay fix** (config): the `ktg_alice` data block used `index: 0`, which the
  resolver rejects (1-based HEPData convention → overlay silently dropped). Changed the 7 `index: 0` →
  `index: 1` (the only `index: 0` in the file; the tables have the ktg distribution as their single
  dependent variable). `ktg_alice` now shows MC + data + ratio. **NOTE: HEPData table `index` is
  1-based here — never write `index: 0`.**

## Disabled-observable audit (2026-06-02): why each non-by-design observable is off

Audit of every disabled observable in `config/STAT_5020.yaml` that is **not**
already an accepted by-design-off case, plus the reason each is off. Data
availability cross-checked against the `hard-sector-data-curation` clone
(`hepdata_database.yaml` registry + `data/5020/<type>/<obs>/` payloads).

### Categories to re-visit
**All ATLAS, z-trigger / gamma-trigger, and v2 (flow) categories need to be
re-visited.** They are disabled wholesale right now; revisit each for correct
inspire/HEPData record, curation, and analyzer support before re-enabling.

### Reasons individual observables are off (analyzer- vs data-blocked)
- **`dijet_trigger_jet/v2_cms`** (CMS, ins2165916) — **analyzer code not available.**
  HEPData *is* curated (CMS HIN-21-002, dijet v₂/v₃/v₄), so this is analyzer-blocked,
  not data-blocked. (Also arguably belongs in the by-design v2-flow bucket.)
- **`inclusive_jet/eec_cms`** (CMS, ins2904406) — **analyzer code not available.**
  HEPData *is* curated (ins2904406-v2); analyzer-blocked, not data-blocked.
- **`inclusive_chjet/pt_mixed_events_alice`** (ALICE) — **preliminary measurement**
  (no inspire/HEPData record yet) **and analyzer code not available.**

### Data-blocked (no curated HEPData → `hepdata: N/A`)
- `inclusive_jet/pt_small_R_atlas` (inspire 2623088), `rg_atlas` (2512925),
  `d12_atlas` (2623088), `dR12_atlas` (2909617) — none present in the curation
  clone or registered in `hepdata_database.yaml`. `d12`/`dR12` are marked
  `# TODO: Update` in the config.
- `dijet_trigger_jet/{pt_pair,xj,yield}_atlas` — see config edit below.

### Config edits made this session
- `z_trigger_hadron/IAA_pt_atlas` → `enabled: false` (the whole `z_trigger_hadron`
  group is now off).
- `dijet_trigger_jet/{pt_pair,xj,yield}_atlas` — `inspire_hep` repointed from the
  **wrong** CMS record (ins2165916) to the correct ATLAS paper
  **inspire 2811406** (arXiv 2407.18796, *"Jet radius dependence of dijet momentum
  balance and suppression in Pb+Pb collisions at 5.02 TeV with the ATLAS
  detector"*); `hepdata: N/A` (not yet curated). `v2_cms` left on ins2165916
  (genuinely CMS).

### Stale curation to clean up
The three ATLAS dijet dirs under
`hard-sector-data-curation/data/5020/dijet_trigger_jet/{pt_pair,xj,yield}_atlas/`
still hold **byte-identical copies of the CMS `ins2165916` payload** (wrong data —
they are not the ATLAS measurement). The config no longer references them; delete
so they aren't mistaken for real ATLAS dijet data.

## Step 4.5 — `charge_cms` encoder migration DONE (2026-06-03)

`inclusive_jet/charge_cms` (jet charge, parametrized by **kappa**) is now migrated end-to-end onto
the observable encoder — analyzer, histogrammer, and plotter all derive the histogram name from
`obs.encode_name_for_storing_in_file(...)`. It previously produced **0 histograms**.

**Root cause (C9):** the hist/plot sub-observable loops keyed on the **pre-migration** YAML name
`"kappa"`, but the schema was renamed to `charge` → the kappa branch never fired → the name was built
without `_k{kappa}` → it never matched the analyzer's columns. (Same class for `zr_alice`: loop keyed
`"r"`, YAML is `subjet_R`; and `alpha`/`angularity_alice` had no branch at all.)

**Changes (5 files):**
- `analyze_events_STAT.py`: charge_cms fill → encoder name `encode_name_for_storing_in_file(jet_R, jet_charge=JetChargeSpec(kappa), tag=...)`, 1D scalar. `jet_pt` (`[120, null]`, single cut bin) is non-essential → omitted. **Dropped the `_unsubtracted` QA companion** (it was 0 hists on the legacy path → no regression).
- `plot_results_STAT_utils.py`: added `("inclusive_jet","charge_cms")` to `ENCODER_MIGRATED_JET_OBSERVABLES`.
- `histogram_results_STAT.py` + `plot_results_STAT.py`: `_encoder_column_name` gained a `charge=` arg (both lockstep copies stay byte-identical); the sub-observable loop renames `"kappa"`→`"charge"` (3-tuple `(label, axis_entry, charge_value)`); per-kappa HEPData table selected via `data_block_params={"jet_charge": JetChargeSpec(kappa).encode()}` (kappa 0.3→Table 8, 0.5→Table 1, 0.7→Table 9). Removed the dead `"r"` branch (lockstep harmonization). Plotter threads `self._encoder_charge` through `get_histogram`.
- `plot_results_STAT_utils.py` `_axis_from_independent_values`: **B11 fix** — charge_cms HEPData tables ship `low`/`high` SWAPPED (`low > high` per bin) → non-monotonic edges → `TAxis::Set` errors. Now normalize lower/upper via `np.minimum/np.maximum` (no-op for well-formed tables), warn once per process (`_warn_once`, "flag for curation"), and a monotonicity guard returns `None` (skip) if edges still aren't strictly increasing.

**No `_raa_denom`:** charge_cms AA `ratio` ≡ `spectra` (same Tables 8/1/9) → `maybe_book_raa_denom` self-skips. It's a self-normalized `1/N dN/dQ` distribution, not a spectrum R_AA.

**Verified e2e (small sample, both arms):** analyzer/histogrammer/plotter all exit 0; 9 non-empty
charge_cms hist keys/arm (3 kappa × pop. centralities/labels); per-kappa binning resolves the right
table; plotter builds matching names 9/9 hits + per-kappa data overlay + MC/data ratio persisted to
`final_results.root`; pp 273 / AA 28 PDFs (+18 / +3 charge_cms vs the Step-4 baseline); **0 TAxis, 0
divide, 0 traceback** errors; Step-4 migrated set (mg_cms/zg_cms/axis_cms/…) counts unchanged.
Reviewed via `/code-review high` — findings addressed (swap-warning + monotonicity guard, lockstep
harmonization, grooming-branch `charge=` symmetry); refuted findings documented.

**⚠️ Curation follow-up (B11):** the swapped `low`/`high` is a **data bug** in the curation repo
(`charge_cms` Tables 1/8/9 + AA ratios); `build_tables.py` also reads `low`/`high` raw. Fix at the
source in the curation track (log in its `CURATION_NOTES.md`); the plotter only papers over it.

**Deferred (carried forward):** `zr_alice` (subjet_R) — uncurated data (A5) **and** loop key-mismatch
(C9); stays off until its HEPData is filled. `angularity_alice` (alpha) — the ungroomed/groomed split
(C7) is **Step 5**; not touched here.

## Step 5 — ungroomed/groomed split DONE (2026-06-03)

The two ALICE charged-jet substructure observables that conflated an ungroomed and a groomed quantity
under one key (edge case C7) are **split into four observables, all on the observable encoder**:

- `mass_alice` → **`mass_alice`** (ungroomed `jet.m()`) + **`mg_alice`** (groomed `m_g`).
- `angularity_alice` → **`angularity_alice`** (ungroomed λ) + **`angularity_groomed_alice`** (groomed λ),
  each parametrized by the **alpha** sub-observable (α = 1, 1.5, 2, 3).

**`double_ratio` was NOT added** (user decision 2026-06-03): mass/angularity are self-normalized
distributions that don't need it, and `pt_y_atlas`'s double-ratio already works via the
`ratio`+self-normalize path (B12). Tracked for a possible later cleanup with Step 7, not Step 5.

**Changes (6 files):**
- `config/STAT_5020.yaml`: split each block into two; each new block declares a SINGLE `grooming_settings`
  entry and partitions its `data:` combinations by grooming (mass pp 33/35/37 vs 34/36/38; AA ratio
  39/41/43/45 vs 40/42/44/46; angularity pp 41–64 and AA ratio 1–32 split ungroomed/groomed). New
  `mg_alice` / `angularity_groomed_alice` blocks added. Stale "includes both groomed and ungroomed"
  comments fixed.
- `jetscape_analysis/analysis/analyze_events_STAT.py`: ungroomed mass + angularity fills migrated from
  legacy 2D `[jet_pt, value]` f-strings to **1D-per-pt-bin encoder** fills (`jet_pt_spec` computed locally
  in the ungroomed method); groomed fills renamed to `mg_alice` / `angularity_groomed_alice`. `_unsubtracted`
  companions dropped (no regression, C8 precedent). Kappa standardized to `1.0` float (F5 latent-bug fix).
- `plot/histogram_results_STAT.py` + `plot/plot_results_STAT.py` (lockstep): **alpha sub-observable loop** —
  sub-observable tuple widened 3→4 `(label, axis_entry, charge_value, angularity_spec)`, a new `"angularity"`
  branch, an `angularity=` arg on both `_encoder_column_name` helpers, and per-alpha `data_block_params`
  (`jet_angularity`). Plotter stashes `_encoder_angularity` for `get_histogram`.
- `plot/plot_results_STAT_utils.py`: all four added to `ENCODER_MIGRATED_JET_OBSERVABLES`.
- (curation repo) `hepdata_database.yaml`: `mg_alice` / `angularity_groomed_alice` registry aliases reusing
  the sibling HEPData records (no new files); logged in `CURATION_NOTES.md`.

**Verified e2e (pp small sample, analyzer→histogrammer→plotter, exit 0):** mass_alice 3 + mg_alice 3
non-empty keys (per-grooming binnings differ 13/10/5 vs 16/11/6); angularity_alice 24 + angularity_groomed_alice
24 (4 α × 3 pt × {distribution + `_raa_denom`}, per-alpha overlays differ); data overlays resolve per
pt/grooming/alpha; **0** "no matching table" / TAxis / "cannot divide" errors; Step-4/4.5 migrated set
(mg_cms 30, charge_cms 9, ktg_alice 2, …) unchanged. Encoder names byte-consistent analyzer↔histogrammer
(incl. `kappa_1.0`). Edge-case mechanics documented in `OBSERVABLE_EDGE_CASES.md` section F.

**AA arm pp-verified only:** all four are `centrality: [[0,10]]`; the PbPb small sample (~10–11%) gives
empty AA MC, so the AA R_AA arm is exercised for code-path correctness but not numerically — full AA
validation rides with **Step 7** (needs a 0–10% sample).

## AA data/MC validation pass — DONE + COMMITTED 2026-06-10 (`d180f12`)

Ran STAT_2760 + STAT_5020 R_AA/substructure through the full render and fixed the AA-canvas
data/MC comparison bugs that surfaced. Render workflow: re-plot on **lxplus** (the 6 GB VM OOMs on
the AA plot step) → publish to the single non-versioned site
https://zhangj.web.cern.ch/jetscape_render/ (inline thumbnail gallery; `publish_render.py`).

Plotter (`plot/plot_results_STAT.py`):
- **Hadron R_AA** draws only the hole-subtracted (physical) curve, not the overlapping unsubtracted
  cross-check (was: two legend entries, one visible band, mismatched color).
- **`skip_AA_ratio`** (self-normalized ALICE substructure: mass/angularity/mg/axis) overlays the
  self-normalized AA *distribution* vs the measured PbPb distribution instead of dividing by pp.
- **`AA_difference`** flag for CMS `Dpt_cms` (`Delta_D = D_PbPb − D_pp`, goes negative): subtract the
  pp reference; reference line at 0; y-range admits negatives.
- **Non-ratio AA plots** auto-scale y to the data+MC content with legend headroom (was fixed `[0,3]`).
- **Log-y twin** (`<name>_logxy.pdf`) for every non-negative AA plot.

Data-overlay resolver (`plot/plot_results_STAT_utils.py`):
- Resolve the HEPData record from the artifact OR the parent `hepdata.record`; **warn** instead of
  silently using `infos[0]`. The silent fallback was drawing **pp data on the PbPb canvas** for
  `axis_alice` (the same table name exists in both the pp and PbPb records). Watch this on any
  multi-record observable.
- For `skip_AA_ratio`, prefer the `spectra` (distribution) artifact, falling back to whichever block
  (`spectra`/`ratio`) actually has wired tables — curators put the PbPb distribution in `ratio` for
  mass/angularity/mg but in `spectra` for axis_alice/charge_cms.

Config: `STAT_5020.yaml` — `skip_AA_ratio` on angularity/angularity_groomed/mass/mg/axis; wired
`axis_alice` `AA.spectra` to the ins2648610 PbPb distribution tables. `STAT_2760.yaml` — `skip_AA_ratio`
on mass/g; `AA_difference` on `Dpt_cms`.

**WTA verified correct** (empirically, FastJet in `/root/stat_local_gcc_v5.2.sif`) — the `axis_alice`
MC/data gap was the pp-overlay bug, not WTA. Known-but-immaterial: the `NegativeEnergyRecombiner` set
before `set_recombination_scheme(WTA_pt_scheme)` is silently discarded (only touches the
`_negative_recombiner` band; benign).

**Remaining small items (axis_alice):** align the MC binning to the 12-bin HEPData grid (it has a
spurious 0.035 split → 13 bins); no HEPData distribution table exists for R0.4 pt40–80 or pt100–140;
cosmetic cleanup of the dead `set_recombiner` lines (`analyze_events_STAT.py:905/1287/1660/3540`).

**NOT in `d180f12` (200 GeV groundwork, deferred to the next pass):** `analyze_events_STAT.py` has the
`self.AA`→`self.is_AA` typo fix + a guard on the WIP 200 GeV pion-trigger block; `tables/200/` holds
untracked 200 GeV AuAu data (`.dat`). Commit these with the 200 GeV work.

## Remaining steps (next sessions) — updated 2026-06-10

> Fuller orchestration plan + history live in the project memory
> `jetscape_migration_orchestration.md`. **Note:** that file is under the HOME-dir project
> (`~/.claude/projects/-afs-cern-ch-user-z-zhangj/memory/`), so a session started in *this* work
> dir may not auto-recall it — read it explicitly, or rely on this section (kept in sync).

Steps 1–4 + render-path fixes E1/E2/E3 + the AA R_AA binning fix + plotter cosmetics + **Step 4.5
`charge_cms`** + **Step 5 (ungroomed/groomed split — mass_alice/mg_alice + angularity_alice/
angularity_groomed_alice)** + **Step 6 (STAT_5020)** are **DONE** (see the dated sections above). STAT_2760 + STAT_5020
R_AA/substructure **validated + AA data/MC fixes committed 2026-06-10 (`d180f12`; see the
"AA data/MC validation pass" section above).**
**NEXT = STAT_200 (AuAu 200 GeV) data + more validation.**

**Step 4.5 — parametrized-observable migration.** Wire the observables the analyzer *computes* but the
histogrammer/plotter don't yet iterate (root cause: sub-observable loops keyed on the pre-migration
names; see "Step 4.5 — DONE" above and edge case C9):
- `inclusive_jet/charge_cms` — jet charge by **kappa** — ✅ **DONE (2026-06-03)**.
- `inclusive_chjet/zr_alice` — by **r** (YAML `subjet_R`) — **deferred**: uncurated `data:` block (A5)
  *and* loop key-mismatch (C9). Needs HEPData curation first, then a `subjet_R` branch.
- `inclusive_chjet/angularity_alice` — by **alpha** — **deferred to Step 5** (needs the ungroomed/groomed
  split, C7).

**Step 5 — ungroomed/groomed split — ✅ DONE (2026-06-03).** See the "Step 5 — ungroomed/groomed split
DONE" section above. `double_ratio` intentionally NOT added (not needed; user decision).

**Step 6 — drop legacy YAML keys.**

**STAT_5020 — ✅ DONE + COMMITTED 2026-06-03 (`c83481f`).** Every observable (enabled + disabled) is
off the legacy ROOT overlay path — **zero** active `hepdata_*` / `HEPData-*.root` keys remain.
- `axis_alice` (enabled): stripped `hepdata_pp`/`hepdata_AA` + per-variant `hepdata_*_dir`/`_gname`;
  the `data:` block (pp.spectra + AA.ratio) already carried the equivalent table mappings. It was the
  last enabled observable on the legacy path, so whole-config renders no longer exclude it.
- Self-norm fix (`plot_results_STAT.py`): added `"axis"` to the self-normalize substring trigger —
  the jet-axis-difference MC (`1/σ_inc dN/dΔR`) was not self-normalized and sat ~0 vs data. Latent
  bug surfaced only now that axis_alice finally renders.
- Disabled `xj_gamma_atlas` / `xj_gamma_cms` / `gamma_trigger_jet/pt_atlas`: stripped legacy keys;
  their (uncurated, empty-table) `data:` skeletons remain. `rg_atlas` left as-is (uncurated, *not* a
  MIXED entry — only a dummy `bins:` placeholder).
- Verified e2e (pp, plotter-only on cached histograms): exit 0, 339 PDFs incl. 12 `axis_alice` with
  MC+data overlaid; all HEPData cached locally (no web fetch). NB: `axis_alice` AA arm needs a
  `[0,10]`-contained hydro file (current 10–11% sample fails the `[0,10]` full-containment bin) → Step 7.

**STAT_5020 — ALICE curation observables (IAA / dphi / zr_alice) — pp-DONE+VALIDATED 2026-06-04 (uncommitted batch).**
These were the deferred "curation+wiring" half of Step 6 (axis_alice above was the wiring-only half).

- **`IAA_pt_alice` + `dphi_alice` (semi-inclusive h+jet recoil).** The observable is the trigger-
  background-subtracted recoil distribution Δ_recoil = (1/N_trig^high)·high − c_ref·(1/N_trig^low)·low.
  The analyzer produces only per-trigger-class yields (`_lowTrigger`/`_highTrigger`); **nothing
  assembled Δ_recoil** and the plot path for `hadron_trigger_chjet` was never even called. Added:
  (1) `histogram_results_STAT.py:_build_semi_inclusive_delta_recoil` — builds Δ_recoil per (R[,pt-bin])
  from the booked high/low + the trigger-pt (N_trig) histogram; dphi reuses the IAA-owned trigger-pt
  hist (same triggers). (2) wired `plot_hadron_trigger_chjet_observables` into `plot_results()` (it
  existed but was never invoked) + a pt-bin loop for dphi. (3) **per-trigger normalization:** the
  1/N_trig ratio cancels the standard xsec/weight_sum scaling, so `_scale_one_histogram` SKIPS the
  standard chain for `hadron_trigger_chjet` (else it double-normalizes ~1e3). (4) curation: IAA
  data: block → Tables 1/2/3 (pp/AA/I_AA, ins2693336, R via index); dphi → per-(R,pt) Tables 4-6 /
  7-9 / 10-12. (5) `c_ref` per R (AA only; pp=1): IAA `[0.90,0.82,0.82]` (Fig-5 *integrated*), dphi
  reuses the same (the differential Fig-5 values are not in HEPData — see edge case A10).
  Verified pp (ratio≈1 where populated). **`min_jet_pt: 20 → 7`** (config) so recoil jets reach the
  data's lowest pt bins — needs the slow re-analysis to take effect (the 20 GeV floor left the lowest
  bins empty; this is the ONLY observables that need jets <20 GeV).
- **`zr_alice` (leading-subjet z_r, subjet_R 0.1/0.2).** Was doubly-blocked (C9 loop key-mismatch + A5
  uncurated). Fixed WITHOUT touching the analyzer (its `_R{R}_r{r}` columns already exist): added a
  **`subjet_R` sub-observable branch** (widened the jet sub-observable tuple 4→5; legacy name path,
  not encoder) in both the histogrammer and plotter, threading `data_block_params={jet_subjet_R: r_X}`
  (NOT gated on is_migrated) for the per-r binning + overlay; added `"zr"` to the self-normalize
  trigger; curated the data: block → Tables 5/8 (pp), 6/9 (AA), 7/10 (PbPb/pp ratio). Verified pp
  (ratio≈1).
- **Adversarial review (2026-06-04) — fixed:** (a) the `hadron_trigger_chjet` scale-skip wrongly
  swallowed `nsubjettiness_alice`'s (STAT_2760) self-normalization → guarded with `and not
  self_normalize`. (b) zr_alice's legacy `maybe_book_raa_denom` call omitted `data_block_params`, so
  the r=0.2 R_AA denominator silently resolved to the r=0.1 ratio table → forwarded the param.
- **Large-sample validation + fixes (2026-06-04, 30 pp + 30 PbPb via Condor, `min_jet_pt: 20→5`).**
  The decisive test exposed (and we fixed) two real bugs the small sample masked:
  (1) **Δ_recoil multi-file aggregation** — the per-trigger (1/N_trig) normalization was done per-file
  in the histogrammer, so `hadd`-summing N files over-counted by **n_files** (IAA was ×30 high).
  **Fix:** moved the Δ_recoil build to the **plotter** (`construct_semi_inclusive_histogram`,
  post-aggregation); the histogrammer now books only the raw high/low/N_trig. (2) **dphi pt-window**
  — dphi HEPData is double-differential `(GeV/c·rad)^-1`; the MC is integrated over the jet-pt
  sub-bin, so divide by that window width (×10 for 20-30 GeV). IAA needs NO such factor (its data is
  Δφ-integrated; ratio≈1 as-is). After both fixes, IAA + dphi ratio≈1 at full stats, low-pt bins fill
  (`min_jet_pt: 5` → IAA down to 7 GeV). Also fixed `truncate_tgraph` to skip (not raise) on an AA
  hist/graph x-mismatch, so the **Step 7 R_AA** render completes (72 PDFs, 10-40% centrality, physical
  suppression for hadron + jet R_AA). See OBSERVABLE_EDGE_CASES G1/G4/G6. Plots: `~/e2e_render_0604`.

**NEXT — STAT_2760, then STAT_200** (both still MIXED/OLD). The migrated analyzer/histogrammer/plotter
**code is energy-agnostic and already done**, so these are config-side sweeps (legacy→`data:` schema
per observable) + any missing HEPData curation — NOT a from-Step-1 code redo.

End state: every observable on one path (`data:` + new encoder + `obs.essential_parameters()`).

**Step 7 — PbPb (AA) R_AA coverage verification across ALL centralities, case-by-case.** All
migration testing so far has used a SINGLE PbPb chunk at one narrow centrality, so the AA/R_AA arm
of every observable is essentially unverified across its real centrality range. This step verifies,
per observable × per configured centrality bin, that the AA MC fills, the R_AA renders, and the
per-centrality `ratio` HEPData overlay matches — distinct from the code-migration steps 1–6.

*Why it's not covered yet (centrality mechanics):* precomputed-hydro sample files are FIXED ~1%-wide
centralities (file 0 = cent 10–11%, … file 29 = 39–40%), and the histogrammer fills a bin `[a,b]`
only on FULL CONTAINMENT (`event_cmin ≥ a AND event_cmax ≤ b`; `histogram_results_STAT.py`
`centrality_accepted`). So the one chunk used in e2e (cent 10–11%) only fills `[10,20]`/`[10,30]`.
Of the 24 enabled STAT_5020 R_AA observables: **12 got ZERO AA stats** (their bins are `[0,10]`
and/or `[30,50]`, neither containing 10–11% — `pt_y_atlas`, `IAA_pt_alice`, `dphi_alice`,
`inclusive_chjet/{angularity,axis,mass,zr,ktg,tg,zg,pt}_alice`, `inclusive_jet/pt_alice`); **12 got
PARTIAL** (only `[10,20]`/`[10,30]`); and 8 bins (`[0,5] [0,10] [5,10] [20,30] [30,40] [30,50]
[40,50] [50,80]`) were never exercised. Even the full 30-file sample only spans 10–40%, so sub-10%
and >40% bins stay empty.

*Step 7 worklist:*
1. Re-run with the FULL 30-file sample (not one chunk) → exercises `[10,20]`/`[20,30]`/`[30,40]`/`[10,30]`
   for the 12 partial-coverage observables (CMS/ATLAS jet + hadron). Cheap; do first.
2. Acquire/generate PbPb samples for the MISSING regions — especially **0–10%** (unblocks the 12
   zero-coverage ALICE substructure/h+jet observables **and `pt_y_atlas`**) and **40–80%**
   (`axis_cms` `[50,80]`, hadron `[40,50]`, CMS `[30,50]`). *Cleaner long-term:* switch to
   **event-based** (`real_time_hydro`) samples spanning 0–100% — one event set fills every bin via
   per-event centrality filtering, instead of stitching fixed-centrality files.
3. Case-by-case per observable × centrality: confirm AA MC fills, `plot_RAA` renders, and the
   per-centrality `ratio` table overlays/aligns. **Motivating case: `pt_y_atlas` AA double-ratio**
   (`[0,10]`-only; its R_AA(|y|)/R_AA(0) vs `raa_doubleRatio` is the actual physics result, never
   filled by the 10–11% sample — see B12).
4. Then repeat for **STAT_2760** (18 enabled AA observables, MIXED) and **STAT_200** (AuAu 200 GeV,
   ~6–7 enabled AA observables, MIXED, incl. γ-trigger) — after their Step-6 sweep.

**Side-items (not blocking):** `ktg_alice` DyG variant — DONE (2026-06-08, C6): histogrammer + plotter
loops now process dynamical_grooming for migrated observables + pp DyG data-block key fixed; validated
e2e on the pp sample. v2/flow + γ-trigger analyzers not implemented (by-design-off). 2760 / 200 swept
(2026-06-08, committed) — full render-validation still pending PbPb samples (Step 7).

## Diagnostic snippets

To find places still using the old YAML schema:

```bash
grep -nE 'block\["(jet_R|soft_drop|axis|kappa)"\]' plot/ jetscape_analysis/
grep -nE '"soft_drop"\]' jetscape_analysis/analysis/analyze_events_STAT.py
```

To list observables that have/don't have a new-style `data:` block:

```python
import yaml
d = yaml.safe_load(open("config/STAT_5020.yaml"))
for cls, obs_dict in d.items():
    if not isinstance(obs_dict, dict): continue
    for obs_name, obs_block in obs_dict.items():
        if not isinstance(obs_block, dict): continue
        has_data = "data" in obs_block
        enabled = obs_block.get("enabled", True)
        print(f"{'+' if has_data else '-'} {'E' if enabled else 'd'} {cls}/{obs_name}")
```
