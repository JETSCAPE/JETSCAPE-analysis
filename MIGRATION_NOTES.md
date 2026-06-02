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

### Follow-up surfaced by the full Step 4 ④ render (pre-existing, NOT yet fixed)

The first complete AA+pp render hit **pre-existing plotter schema-rename
casualties** (separate from the encoder migration): `init_common_settings`
still reads the legacy top-level `eta_cut`/`eta_cut_R`/`y_cut`, but the new YAML
nests them as `hadron.eta` / `jet.eta` / `jet.y` → `scale_histogram` raises
`AttributeError: 'eta_cut'`/`'y_cut'` on the hadron + inclusive-jet RAA paths.
Same class as the renames in "What was fixed" #3, just missed for `eta`/`y`. Fix
= read the nested values in `init_common_settings`. Tracked separately (user,
2026-06-02); see `OBSERVABLE_EDGE_CASES.md` E1.

## Recommended order for next session

1. ~~**Fix `bins_from_config`**~~ — DONE, see "Step 1" above.
2. **Run end-to-end on the small sample** and confirm non-groomed
   histograms appear. Sanity-check shapes against a known-good 2.76 TeV
   reference if you have one.
3. **Migrate the rest of the analyzer to the new encoder** observable
   family at a time (hadron RAA first — most heavily measured). Keep
   the 1D-per-bin convention I started for the groomed observables.
4. ~~**Migrate the histogrammer** to the encoder names~~ — DONE (Step 4 ③,
   `5a4f30f`); see "Step 4 — DONE" above. (The deeper `essential_parameters()`
   walk + declarative `combinations:` + `double_ratio` dispatch is still future.)
5. ~~**Migrate the plotter** the same way~~ — DONE (Step 4 ④, `47583d2`).
6. **Cleanup items** (`mass_alice` split = Step 5; strip legacy `hepdata_*` keys
   incl. `axis_alice` + `double_ratio` type + bin source = Step 6; plus the E1
   `eta`/`y` schema-rename render fix above).

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
