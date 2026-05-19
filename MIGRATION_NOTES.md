# Observable-encoding migration — work in progress

Raymond started an in-flight migration on `dev-aggregation`. This file
documents the state as of 2026-05-19, what was fixed to keep the pipeline
running on the new YAML, and the work that remains to complete the
end-to-end conversion.

The user-facing read of this is: **the analyzer now completes; the
histogrammer runs but silently produces no histograms for most observables
because the binning lookup hasn't been migrated yet.** Plan for finishing
the migration is at the end.

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
- elif `"hepdata"` in block → read from the HEPData ROOT file referenced

In the new YAML neither key is present on most observables (3 still have explicit `bins:`, 3 still have a top-level `hepdata:` filename). The other ~49 observables have only the new `data:` block, which references HEPData by `inspire_hep_id` + `table` + `index`. The histogrammer doesn't yet know how to consume that.

**Possible paths to fix:**
- Quick: hand-add explicit `bins:` to every observable block (tedious, defeats Raymond's design).
- Right: extend `bins_from_config` to read from the new `data:` block via the `data_curation` module, which already parses HEPData v2 records and exposes binning. See `jetscape_analysis/data_curation/data.py:parse_binning_block` (~line 469) and `jetscape_analysis/data_curation/hepdata_utils.py`.

Without this fix, after running through the analyzer + histogrammer, the resulting ROOT file will contain zero (or near-zero) histograms despite the analyzer doing all the work correctly.

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

## Recommended order for next session

1. **Fix `bins_from_config`** to read binning from the `data:` block (or
   directly from the HEPData submodule). Without this, no histograms come
   out regardless of any further work. Smallest change with biggest
   payoff.
2. **Run end-to-end on the small sample** and confirm non-groomed
   histograms appear. Sanity-check shapes against a known-good 2.76 TeV
   reference if you have one.
3. **Migrate the rest of the analyzer to the new encoder** observable
   family at a time (hadron RAA first — most heavily measured). Keep
   the 1D-per-bin convention I started for the groomed observables.
4. **Rewrite the histogrammer** to walk `obs.essential_parameters()`
   instead of hand-looping. Replace the "Custom skip" exceptions with
   declarative `combinations:` from the YAML. Add `spectra` / `ratio` /
   `double_ratio` dispatch.
5. **Rewrite the plotter** the same way — much smaller code path than
   the histogrammer, similar shape.
6. **Cleanup items** (`mass_alice` split, double_ratio type, bin source).

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
