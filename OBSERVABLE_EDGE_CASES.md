# Observable edge cases

A living registry of edge cases in the observable / HEPData handling, and what
the code does about each. Companion to `MIGRATION_NOTES.md`.

**Guiding principle.** In the ideal/production case — full input statistics
covering every centrality, and every analyzer implemented — the JETSCAPE **MC
(analyzer)** should fill a histogram for every enabled observable. An
empty/absent MC histogram is therefore **not** expected in general. In our
small-sample tests it does happen, for two benign MC-side reasons tracked
separately from this doc:
- the test sample spans only a limited centrality range (e.g. the PbPb small
  sample is ~10–20%, so observables defined only for other centralities don't
  fill), and
- some analyzers are not yet implemented (notably **v2/flow** and
  **γ-triggered** observables).

This document instead catalogs the **HEPData-side** edge cases: the measured-data
binning and the data overlay come from HEPData YAML, and not every
observable/centrality/pt combination exists (or exists in the same shape) there.
Where HEPData is missing or oddly shaped, the binning/overlay code either
**falls back**, **protects** (synthesizes/sanitizes), or **skips** (no histogram
= null/absent, which is acceptable *for the comparison* — it just means no data
to compare that MC against). Each entry states which.

This follows Raymond Ehlers' migration design (the `data:` block / `observable`
encoder as the single source of truth); Raymond's guidance is the authority for
how these cases should be handled.

Scope so far: audited/verified on **`config/STAT_5020.yaml`** (the arm we run
end-to-end). `STAT_2760.yaml` / `STAT_200.yaml` are **not yet audited** — see
"Open / to-audit" at the end.

Legend for **Behaviour**: `fallback` = uses an alternative HEPData artifact;
`protect` = sanitizes/synthesizes so it still works; `skip(null)` = no histogram
/ no overlay produced (acceptable); `pending` = known-unhandled, tracked for a
later step.

---

## A. HEPData availability / structure (binning + data overlay)

These are resolved in `plot/plot_results_STAT_utils.py` —
`_resolve_data_hepdata_table` (line 195), used by both `bins_from_data_block`
(282, the histogram binning) and `tgraph_from_data_block` (460, the measured-data
overlay). Artifact selection: **AA → `ratio` (R_AA), pp → `spectra`**, with a
fallback to the other when the preferred one is absent.

Enabled-observable inventory below is for **5020** (audited 2026-06-01). "none
enabled" means no currently-enabled 5020 observable hits that case — the branch
is a guard for malformed/partial HEPData or for future/other-√s configs.

| # | Edge case | Observable(s) — 5020, enabled | Behaviour | Where |
|---|---|---|---|---|
| A1 | **No `ratio` for AA** (only `spectra`) | none enabled (the disabled `gamma_trigger_jet/Dz_atlas`,`Dpt_atlas` are `AA-no-ratio`) | **fallback** → AA uses `spectra` binning/overlay | utils:217-219 (`artifact = ratio if is_AA else spectra; if not artifact: the other`) |
| A2 | **No `spectra` for pp** (only `ratio`) | none enabled | **fallback** → pp uses `ratio` | utils:217-219 |
| A3 | **pp-only observable** (no AA block) | none enabled | AA request → fallback finds nothing → `skip(null)`; pp works normally | utils:217-221 |
| A4 | **AA-only observable** (no pp block) | none enabled | pp request → fallback finds nothing → `skip(null)`. Consequence: no MC pp reference, so the AA MC R_AA (`plot_RAA` divides by `h_pp`, plot_results_STAT.py:1105-1108, skips at :1121-1123 if `h_pp` missing) can't be formed → that observable's RAA plot is skipped. (Data overlay R_AA still fine — it's pre-ratioed in HEPData.) | utils:217-221 |
| A5 | **`data:` block has spectra/ratio structure but the `table:`/`index:` references are empty/unfilled** (not yet curated) | **`inclusive_chjet/zr_alice`**, **`hadron_trigger_chjet/IAA_pt_alice`**, **`hadron_trigger_chjet/dphi_alice`** (all enabled; pp `spectra` + AA `spectra`+`ratio` present but the table entries are blank) | `skip(null)` — expands to no entry with a non-empty `table` → "No matching data table"; MC fills, no histogram produced | utils:256-258 |
| A6 | **No `data:` block at all / not in HEPData DB** (and no old `hepdata:`/`bins:`) | **`z_trigger_hadron/IAA_pt_atlas`** (enabled, NO-DATA-BLOCK) | binning: `skip(null)` ("No binning found"); overlay: `data_distribution=None`. MC fills nothing for it. | `bins_from_config` (utils:67-68); DB miss → utils:212 |
| A7 | **`data:` block + tables present, but no table matches the specific (centrality, jet_R, jet_pt) the analyzer fills** — HEPData doesn't cover every MC combination | **`inclusive_jet/zg_cms`** (high-pt bins), **`inclusive_jet/axis_cms`** (also unmigrated, see C4) | `skip(null)` per-combination + "No matching data table …". MC produced those combos; they simply aren't histogrammed. | utils:256-258 |
| A8 | **Old `hepdata:` ROOT path, not the `data:` block** — `bins_from_config` checks `"hepdata" in block` *before* `"data" in block`, so an observable carrying the legacy `hepdata:`/`hepdata_*_hname` keys (even if it also has a `data:` block) uses the ROOT-file lookup | **`gamma_trigger_jet/xj_gamma_atlas`**, **`gamma_trigger_jet/xj_gamma_cms`** (enabled; kept on the legacy path) | **fallback (legacy)**: binning via `bins_from_hepdata` (ROOT), overlay via `tgraph_from_hepdata` (ROOT). Needs the HEPData `.root` file under `data/STAT/5020/...`. | `bins_from_config` (utils:59-61), `bins_from_hepdata` (utils:73) |

**Note on A1/A2 ("only spectra available" case):** the fallback means an
observable that only ships one artifact still gets binned and overlaid from
whatever it has; nothing special is required of the user.

---

## B. HEPData table content (within a matched table)

Handled in `_axis_from_independent_values` (utils:153) and the per-point loop of
`tgraph_from_data_block`.

| # | Edge case | Observable(s) — 5020, enabled | Behaviour | Where |
|---|---|---|---|---|
| B1 | **Bin edges given as a center `value`** instead of `{low, high}` (point/figure data) | **`inclusive_chjet/pt_alice`** (ML/AB charged-jet R_AA, "Figure 4c") — the only one | **protect**: synthesize edges as midpoints between centers; `centers` = bin centers (so histogram bin centers coincide with overlay x-points → they stay aligned) | utils:159-168 |
| B2 | **Single center / non-monotonic centers** (can't infer a bin width, or would make non-increasing edges that break `TH1F`) | **none** — defensive guard only (no enabled 5020 observable triggers it) | `skip(null)` — `_axis_from_independent_values` returns `None` | utils:166 |
| B3 | **Integer bin edges** in the YAML | **`inclusive_jet/pt_alice`, `pt_atlas`, `pt_y_atlas`, `pt_cms`** | **protect**: forced to `float64` (ROOT `TH1F(const double* xbins)` needs a double buffer) | utils:176-177, 282-296 |
| B4 | **Unusable dependent value** — missing (`value` is `None`/`""`/`"-"`) or non-numeric (not `float`-able) | **`inclusive_jet/pt_cms`, `inclusive_jet/mg_cms`** (tables contain some empty/`-` entries) | **protect**: that point is skipped in the overlay (two guards: the `val is None …` continue and the `except (TypeError, ValueError)` continue) | tgraph_from_data_block (per-point loop) |
| B5 | **Empty / missing `independent_variables`** | **none** — defensive guard (malformed/partial table) | `skip(null)` + warning | bins_from_data_block (axis `None` guard) |
| B6 | **HEPData YAML referenced in the DB but not present on disk** (submodule not fully unpacked) | **none currently** — all referenced files present in the sibling clone; environmental guard | `skip(null)` + `logger.warning("HEPData YAML missing: …")` | utils (yaml_path `.exists()` guard) |
| B7 | **Multiple HEPData records / `record_id` not found** | **`inclusive_chjet/axis_alice`, `inclusive_chjet/angularity_alice`** (2 records each in the DB) | **protect**: match on `inspire_hep_id`; fall back to the first record | utils (`next((i … ), infos[0])`) |
| B10 | **`independent_variables` / `dependent_variables` length mismatch** (more dependent values than bins) | **none** — defensive guard | **protect**: the overlay loop stops at the shorter list (`if i >= len(centers): break`); excess dependent points dropped | tgraph_from_data_block (per-point loop) |

### Observable-specific binning special cases (HEPData omits/keeps a bin)
| # | Edge case | Behaviour | Where |
|---|---|---|---|
| B8 | `zg_alice`, `tg_alice` — HEPData includes a leading "untagged" bin we treat as underflow | **protect**: drop the leading bin (`bins[1:]`) in BOTH binning and overlay | utils:134/299 (bins), utils:530 (tgraph) |
| B9 | `pt_y_atlas` — HEPData omits the `0<|y|<0.3` bin (it's the R_AA denominator, ≡1) | **protect (asymmetric)**: binning **inserts** a `0.0` edge so the MC histogram has that bin; the overlay does **not** add a point there (HEPData has no measurement). Result: histogram has N+1 bins, overlay has N points — matches the old ROOT behaviour; `truncate_tgraph`'s offset logic handles the mismatch. **Documented asymmetry, not a bug.** | bins: utils:137/303; tgraph: no insert (only the zg/tg drop at 530) |

---

## C. Analyzer / MC-side groomed & substructure edge cases

| # | Edge case | Behaviour | Where |
|---|---|---|---|
| C1 | **Untagged groomed jet** (Soft Drop found no splitting → `Delta() < 0`, empty groomed pair). `fjext.lambda_beta_kappa` segfaults on the empty pair (the scalar `.kt()/.Delta()/.z()` return negative sentinels safely; only the constituent-iterating `lambda_beta_kappa` crashes). | **protect**: only compute groomed angularity when `Delta() >= 0`; otherwise emit `-1.0` → underflow (matches the ktg/zg/theta_g sentinel convention) | analyze_events_STAT.py:1315-1318 (`angularity_alice`) |
| C2 | **GroomerShop lifetime / use-after-free** — the reclustering `ClusterSequence` was freed on function return while the groomed jet's constituents still pointed at it (data-dependent segfault; pp small sample hit it, PbPb didn't). | **protect (fixed)**: `calculate_groomed_jet` now returns the `GroomerShop`; callers keep it alive while accessing groomed constituents | analyze_events_base_STAT.py:847 + the two callers in analyze_events_STAT.py |
| C3 | **`axis_alice` data shape** — was 2D `[jet_pt, deltaR]` but the new encoder name already pins the pt bin (redundant / mis-splits). | **migrated**: now 1D-per-pt-bin (scalar `deltaR`), passing essential `jet_R`,`jet_pt`,`jet_axis` (grooming carried inside `jet_axis`, so `jet_grooming_settings` is not passed) | analyze_events_STAT.py:1237-1287 |
| C4 | **`axis_cms` — the ANALYSIS CODE itself is not migrated** (distinct from C5): the analyzer fill hand-builds the f-string `inclusive_jet_axis_cms_R{jetR}_WTA_Standard{tag}` and appends 2D `[jet_pt, deltaR]` — it never calls the encoder. (Contrast C3 `axis_alice`, and `mg_cms`/`zg_cms` whose ANALYZER is migrated.) So for axis_cms **both the analyzer and the histogrammer are unmigrated.** It also currently fails the `data:` lookup (A7) → no histogram. | **pending (Step 4 ③)** — not on the pp critical path (doesn't crash). Migration must add `jet_pt` (essential) and 1D-per-bin, like C3. | analyzer fill: analyze_events_STAT.py:1603-1622 |
| C5 | **Groomed observables dropped by the HISTOGRAMMER/PLOTTER** (analyzer side IS migrated) — the analyzer writes new **encoder** names (e.g. `inclusive_jet_mg_cms_jet_pt_140.0_160.0_jet_R_0.4_jet_grooming_settings_…`) but the histogrammer/plotter still hand-build **old f-strings**, so they never match → `mg_cms`/`zg_cms`/etc. populate on the analyzer side but are silently dropped when histogramming/plotting. Verified: analyzer writes 24 `mg_cms` columns, histogrammer produces 0 `mg_cms` keys. | **pending — THIS IS THE NEXT STEP (Step 4 ③ histogrammer, then ④ plotter).** Fix: have hist/plot derive the column name from `obs.encode_name_for_storing_in_file(...)` (kwargs filtered by `obs.essential_parameters()`) instead of the f-strings. | hist/plot groomed loops |

---

## D. Histogrammer robustness (null / invalid output)

| # | Edge case | Behaviour | Where |
|---|---|---|---|
| D1 | **Degenerate binning** (`bins` is `None`, empty, or a single edge → `TH1F(…, len(bins)-1, …)` with ≤0 bins) | **protect**: `histogram_1d_observable` / `histogram_2d_observable` return early if `len(bins) < 2` | histogram_results_STAT.py:1029, 1062 |
| D2 | **ROOT `gDirectory` orphaning** — booked histograms were owned by whichever HEPData `TFile` was open (for bin lookups) and got deleted when it closed → dangling/null entries in `output_list` (165 in AA, 55 in pp before the fix). | **protect (root cause fix)**: `ROOT.TH1.AddDirectory(False)` so new histograms aren't tied to a directory; everything is written explicitly | histogram_results_STAT.py:29 |
| D3 | **Any residual null/non-writable output object** | **protect (defense-in-depth)**: `write_output_objects` skips null entries with a single summary warning instead of crashing the whole file | histogram_results_STAT.py:1154, 1168 |

---

## Open / to-audit

- **`STAT_2760.yaml` and `STAT_200.yaml`** not yet audited for the A/B edge cases — repeat the `data:`-structure scan there.
- **A7 (no matching table for a filled MC combination)** — the full per-(observable, centrality, pt) list of skipped combinations hasn't been enumerated exhaustively; representative observables are noted. If a *physics* comparison is expected for one of these, it indicates a missing HEPData table (push to the experiment) rather than a code bug.
- **C4/C5 (axis_cms + histogrammer/plotter encoder-name migration)** — tracked in `MIGRATION_NOTES.md` as Step 4 ③/④; until then groomed observables populate on the analyzer side but not in the histograms/plots.
