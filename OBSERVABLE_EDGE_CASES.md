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
| A5 | **`data:` block has spectra/ratio structure but the `table:`/`index:` references are empty/unfilled** (not yet curated) | **`inclusive_chjet/zr_alice`**, **`hadron_trigger_chjet/IAA_pt_alice`**, **`hadron_trigger_chjet/dphi_alice`** (all enabled; pp `spectra` + AA `spectra`+`ratio` present but the table entries are blank) | `skip(null)` — expands to no entry with a non-empty `table` → "No matching data table"; MC fills, no histogram produced. **`zr_alice` is doubly blocked** — its sub-observable loop is also mis-keyed (`"r"` vs YAML `subjet_R`, see C9). It stays **deferred** until BOTH are resolved (curate the HEPData, then re-add a `subjet_R` branch to the hist/plot loops). | utils:256-258 |
| A6 | **No `data:` block at all / not in HEPData DB** (and no old `hepdata:`/`bins:`) | **`z_trigger_hadron/IAA_pt_atlas`** (enabled, NO-DATA-BLOCK) | binning: `skip(null)` ("No binning found"); overlay: `data_distribution=None`. MC fills nothing for it. | `bins_from_config` (utils:67-68); DB miss → utils:212 |
| A7 | **`data:` block + tables present, but no table matches the specific (centrality, jet_R, jet_pt) the analyzer fills** — HEPData doesn't cover every MC combination | **`inclusive_jet/zg_cms`** (high-pt bins), **`inclusive_jet/axis_cms`** (also unmigrated, see C4) | `skip(null)` per-combination + "No matching data table …". MC produced those combos; they simply aren't histogrammed. | utils:256-258 |
| A8 | **Old `hepdata:` ROOT path, not the `data:` block** — `bins_from_config` checks `"hepdata" in block` *before* `"data" in block`, so an observable carrying the legacy `hepdata:`/`hepdata_*_hname` keys (even if it also has a `data:` block) uses the ROOT-file lookup | **`gamma_trigger_jet/xj_gamma_atlas`**, **`gamma_trigger_jet/xj_gamma_cms`** (enabled; kept on the legacy path) | **fallback (legacy)**: binning via `bins_from_hepdata` (ROOT), overlay via `tgraph_from_hepdata` (ROOT). Needs the HEPData `.root` file under `data/STAT/5020/...`. | `bins_from_config` (utils:59-61), `bins_from_hepdata` (utils:73) |

| A9 | **One HEPData table per grooming / axis variant** — for observables where `jet_grooming_settings` or `jet_axis` is essential (`mg_cms`: z_cut 0.1 vs 0.5,β1.5; `axis_alice`: WTA_Standard vs two WTA_SD groomings), the `data:` block has a *separate* table per variant. The original `_matches` only constrained centrality/jet_R/jet_pt → it picked the **first** matching table, so every grooming/axis variant of the same (centrality,jet_R,pt) shared the wrong table's binning/overlay. | **protect (Step 4 ③, 2026-06-02)**: `_resolve_data_hepdata_table` gained a `data_block_params` arg; the histogrammer passes `{jet_grooming_settings: GroomingSettingsSpec(...).encode()}` (WITH the `SD_`/`DyG_` prefix — the data-block encoding, distinct from the analyzer's storage encoding) or `{jet_axis: JetAxisDifferenceSpec(...).encode()}`, merged into `desired`. Verified: `mg_cms` z=0.1→Figure 2a, z=0.5→Figure 2b; `axis_alice` WTA_Standard→Table 14, WTA_SD z0.2→Table 19, WTA_SD z0.1→Table 15 (each its own table, vs all→first table before). Entries that don't constrain the key impose no condition (see `_matches`), so non-keyed observables are unaffected. **Step 4 ④ (2026-06-02, `47583d2`)** extended the same `data_block_params` threading to the **overlay** path (`tgraph_from_data_block`, used by the plotter), so the measured-data overlay now also picks the right per-variant table. **Caveat:** `axis_alice` still carries legacy `hepdata_pp`/`hepdata_AA` keys, so its overlay routes through `tgraph_from_hepdata` (ROOT) and ignores `data_block_params` (both WTA_SD variants share one ROOT graph + the `.root` file isn't on disk); fixing this means stripping those legacy keys so it routes through `data:` — deferred to Step 6. | utils:`_resolve_data_hepdata_table` (desired update), threaded via `bins_from_config`/`bins_from_data_block`/`tgraph_from_data_block` |

| A10 | **Semi-inclusive `Δ_recoil` `c_ref` is a paper constant, NOT in HEPData** — the per-trigger-class underlying-event correction `c_ref` used in the AA recoil subtraction `Δ_recoil = (1/N_trig^high)·high − c_ref·(1/N_trig^low)·low` is reported **only in a paper figure** (Fig 5 of the PRC `ins2693247`), never in a HEPData table, so it cannot be auto-curated and must be hardcoded per jet R in the config. | **`hadron_trigger_chjet/IAA_pt_alice`** (this IS the **integrated** analysis → `c_ref: [0.90, 0.82, 0.82]` for R=0.2/0.4/0.5, the Fig-5 integrated values); **`hadron_trigger_chjet/dphi_alice`** (the **Δφ-differential** analysis → Fig 5's differential `c_ref` is **not in HEPData**, so by user decision 2026-06-04 we **reuse the integrated `[0.90, 0.82, 0.82]` for every jet-pt bin**). | **manual config value**: `_build_semi_inclusive_delta_recoil` reads `block["c_ref"][R_index]` **only on the AA arm** (`is_AA`); **pp uses `c_ref = 1.0`** (negligible underlying-event bias) so **pp renders are unaffected** — this only matters once the AA arm is run. The code applies one `c_ref` per R; a genuinely per-(R, jet-pt) differential `c_ref` (if the values ever surface) would need a code change. Also note `[0.90,0.82,0.82]` vs the legacy/“production” `[0.99,0.96,0.93]` (TG3) discrepancy is unresolved. | config `c_ref:` (per observable); `histogram_results_STAT.py:_build_semi_inclusive_delta_recoil` |

| A11 | **Published AA spectra carry per-bin DISPLAY scale factors** — HEPData sometimes stores values exactly as drawn in the paper figure, with an offset baked into each dependent-variable column so the curves separate visually. Those values are NOT comparable to MC, and the framework has no mechanism to undo a per-bin multiplicative factor. | **`inclusive_jet/rg_atlas`** (enabled 2026-08-06): Tables 7-10 (Figure 9, PbPb per-event r_g yields) label their pt columns `158 < pT < 200 GeV(x0.05)`, `200 < pT < 315 GeV(x0.5)`, `315 < pT < 501 GeV(x5)`. The pp companion (Table 2) has **no** such factors. | **skip (deliberate)**: `AA.spectra` left with `table: ""` plus a comment recording why. The R_AA tables (15-18) are unscaled, so the AA arm bins from `ratio` as normal and nothing is lost — R_AA *is* the measurement. **Check for this whenever wiring an `AA.spectra` block**: grep the dependent-variable qualifiers for `(x` before trusting the values. | config `data.AA.hepdata.spectra` (blank by design) |

| A12 | **Record must also be registered in `hepdata_database.yaml`** — `_resolve_data_hepdata_table` looks up `f"{sqrts}/{observable_type}/{observable}"` in the curation DB *before* touching the config's `data:` block. A missing entry returns `(None, None)` behind a single `No HEPData database entry` warning, so a perfectly-curated `data:` block silently yields no bins and no overlay. | any newly-added record (hit while curating **`rg_atlas`**, 2026-08-06) | **manual step**: add `{sqrts}/{class}/{obs}:` with `directory` / `inspire_hep_id` / `version` / `tables_to_filenames` (HEPData table name → data file). **AND mind which clone is read:** `BASE_DATA_DIR` prefers the **submodule** `JETSCAPE-analysis/data/hard-sector-data-curation/`, falling back to the sibling clone only when the submodule has no DB file. The submodule has been populated since 2026-06-09, so **edits made only to the sibling clone are invisible to the code** — update both. | `hepdata_utils.py:26-30` (BASE_DATA_DIR); `plot_results_STAT_utils.py:273` (DB lookup) |

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
| B11 | **Bin boundaries with `low`/`high` SWAPPED** (`low > high` per bin) — a curation data bug | **`inclusive_jet/charge_cms`** (Tables 1/8/9 + their AA ratio tables: every Q bin stored as `{low: x+Δ, high: x}`) | **protect**: `_axis_from_independent_values` normalizes lower/upper via `np.minimum/np.maximum` so edges stay monotonic (a well-formed `low < high` table is a no-op), warns **once per process** (`_warn_once`, "flag for curation"), and a follow-up monotonicity guard returns `None` (→ `skip(null)`) if the resulting edges still aren't strictly increasing (e.g. reversed/overlapping rows). **Underlying data bug → fix in the curation track** (see note below). | utils:`_axis_from_independent_values` |

### Observable-specific binning special cases (HEPData omits/keeps a bin)
| # | Edge case | Behaviour | Where |
|---|---|---|---|
| B8 | `zg_alice`, `tg_alice` — HEPData includes a leading "untagged" bin we treat as underflow | **protect**: drop the leading bin (`bins[1:]`) in BOTH binning and overlay | utils:134/299 (bins), utils:530 (tgraph) |
| B9 | `pt_y_atlas` — HEPData omits the `0<|y|<0.3` bin (it's the R_AA denominator, ≡1) | **protect (asymmetric)**: binning **inserts** a `0.0` edge so the MC histogram has that bin; the overlay does **not** add a point there (HEPData has no measurement). Result: histogram has N+1 bins, overlay has N points — matches the old ROOT behaviour; `truncate_tgraph`'s offset logic handles the mismatch. **Documented asymmetry, not a bug.** | bins: utils:137/303; tgraph: no insert (only the zg/tg drop at 530) |
| B12 | `pt_y_atlas` — **|y| double-ratio; no pp data table (2026-06-03 fix).** The observable is the ATLAS \|y\| double-ratio **R_AA(\|y\|)/R_AA(\|y\|<0.3)** (the rapidity-dependence of the PbPb jet suppression). It is built in two steps: each arm is first **self-normalized to its own** `|y|<0.3` bin (`h.GetBinContent(1)`; `plot_results_STAT.py:1192-1209`), then **`plot_RAA` divides the AA self-ratio by the pp self-ratio** (`:1390`), i.e. final = (AA(\|y\|)/AA(0)) / (pp(\|y\|)/pp(0)) = R_AA(\|y\|)/R_AA(0) — so the pp MC **does** enter, as the R_AA denominator. What is missing is pp **data**: the only \|y\| HEPData table (`raa_doubleRatio_c0_y0.yaml`, ind. var `ABS(YRAP)`, 5 bins) is **PbPb**; the pp HEPData (`ppCrossX_y0.yaml`) is a **pT spectrum** (ind. var `PT [GeV]`, 14 bins) — a different axis. Two symptoms: (a) the MC \|y\| values (analyzer fills 2D `[jet_pt, \|y\|]`) must bin on the \|y\| edges, else they collapse into the first pT bin and the self-ratio zeroes the pp MC; (b) the pp arm has no pp \|y\| data, so loading the pp `data:` (pT) overlay drew a misleading off-axis "Data" legend. | **protect**: (a) histogrammer forces the \|y\| (`ratio` table) binning for the MAIN histogram in BOTH arms (`bins_is_AA = self.is_AA or observable == "pt_y_atlas"`) so the pp MC self-ratio is the correct 5-bin \|y\| shape (verified non-zero: integrals 2.26/3.33/2.09/1.86 for pt0–3, was 0) AND is binning-matched to the AA arm for `plot_RAA`'s divide (the `_raa_denom` is skipped by the spectra==ratio equality guard). (b) config sets **`skip_pp: true`** → prints *"skip data plot -- no pp data in HEPData"*; and `init_observable` now **gates the `data:` overlay** off for the pp arm of any `skip_pp` observable (`elif "data" in block and not (self.skip_pp and not self.is_AA)`) so no spurious "Data" legend. **Analysis-only fix; no curation change.** The real data comparison is the AA arm (R_AA(\|y\|) double-ratio vs `raa_doubleRatio`), **not exercisable in this sample**: pt_y_atlas is `[0,10]`-centrality-only and the available PbPb chunk is `[10,11]` (AA parquet has no pt_y_atlas column). | histogram_results_STAT.py (`bins_is_AA`); plot_results_STAT.py (`init_observable` skip_pp gate); config `skip_pp: true` |

---

## C. Analyzer / MC-side groomed & substructure edge cases

| # | Edge case | Behaviour | Where |
|---|---|---|---|
| C1 | **Untagged groomed jet** (Soft Drop found no splitting → `Delta() < 0`, empty groomed pair). `fjext.lambda_beta_kappa` segfaults on the empty pair (the scalar `.kt()/.Delta()/.z()` return negative sentinels safely; only the constituent-iterating `lambda_beta_kappa` crashes). | **protect**: only compute groomed angularity when `Delta() >= 0`; otherwise emit `-1.0` → underflow (matches the ktg/zg/theta_g sentinel convention) | analyze_events_STAT.py:1315-1318 (`angularity_alice`) |
| C2 | **GroomerShop lifetime / use-after-free** — the reclustering `ClusterSequence` was freed on function return while the groomed jet's constituents still pointed at it (data-dependent segfault; pp small sample hit it, PbPb didn't). | **protect (fixed)**: `calculate_groomed_jet` now returns the `GroomerShop`; callers keep it alive while accessing groomed constituents | analyze_events_base_STAT.py:847 + the two callers in analyze_events_STAT.py |
| C3 | **`axis_alice` data shape** — was 2D `[jet_pt, deltaR]` but the new encoder name already pins the pt bin (redundant / mis-splits). | **migrated**: now 1D-per-pt-bin (scalar `deltaR`), passing essential `jet_R`,`jet_pt`,`jet_axis` (grooming carried inside `jet_axis`, so `jet_grooming_settings` is not passed). **NOTE (Step 4 ③):** this covers only the *WTA_SD* (groomed) axis entries; the *WTA_Standard* (ungroomed) entry had a separate un-migrated 2D f-string fill — now also migrated, see C4. | analyze_events_STAT.py (groomed fill) |
| C4 | **`axis_cms` + `axis_alice` WTA_Standard — the ANALYZER fills hand-built f-strings + 2D `[jet_pt, deltaR]`** (axis_cms: `inclusive_jet_axis_cms_R{jetR}_WTA_Standard{tag}`; axis_alice WTA_Standard: `inclusive_chjet_axis_alice_R{jetR}_WTA_Standard{tag}`) — they never called the encoder, so the (now encoder-based) histogrammer couldn't find them. | **migrated (Step 4 ③, 2026-06-02)**: both now emit the encoder name + 1D scalar `deltaR` per pt bin (`jet_R`,`jet_pt`; plus `jet_axis` for axis_alice where it's essential, omitted for axis_cms where the single WTA_Standard variant is non-essential). axis_cms still has no *pp* HEPData table (A-class) so it stays empty in pp; it populates in AA. | analyzer ungroomed-jet fills |
| C5 | **Groomed observables dropped by the HISTOGRAMMER** (analyzer side IS migrated) — the analyzer writes **encoder** names (e.g. `inclusive_jet_mg_cms_jet_pt_140.0_160.0_jet_R_0.4_jet_grooming_settings_z_cut_010_beta_0`) but the histogrammer hand-built **old f-strings** → `mg_cms`/`zg_cms`/etc. populated on the analyzer side but were silently dropped when histogramming. | **DONE (Step 4 ③, 2026-06-02)**: the histogrammer now derives the storage column / histogram name from `obs.encode_name_for_storing_in_file(...)` for the migrated set (`mg_cms,zg_cms,rg_atlas,axis_cms,ktg_alice,zg_alice,tg_alice,axis_alice`) via a `_encoder_column_name` helper + `ENCODER_MIGRATED_JET_OBSERVABLES` dispatch. **Subtlety:** the analyzer passes the *underlying* `SoftDropSpec` (no `SD_` prefix, e.g. `z_cut_010_beta_0`) as `jet_grooming_settings`, so the helper uses `convert_to_grooming_method_spec` (NOT `GroomingSettingsSpec`) to match byte-for-byte. Verified: groomed keys populate in both AA and pp, non-groomed unchanged (AA 47/47, pp 59/59). **PLOTTER: DONE (Step 4 ④, 2026-06-02, `47583d2`)** — the plotter builds the migrated histogram name from the same encoder via its own `_encoder_column_name` (a byte-for-byte **lockstep copy** of the histogrammer's; keep the two in sync — candidate for a shared helper in the eventual encoder-consolidation refactor) + a `get_histogram` branch on `self._is_migrated_obs` (encoder context stashed in the `plot_jet_observables` loop; pt suffix suppressed on the name, kept on binning/overlay lookups; collection_label folded into the encoder `tag`). Verified the built names resolve **exactly** against the histogrammer output (AA mg_cms 6 / zg_cms 3 / axis_cms 12; pp mg_cms 24 / zg_cms 3 / ktg·zg·tg_alice 2 / axis_alice 6, the 3 axis variants incl. both WTA_SD distinguished); non-migrated unaffected; rg_atlas skipped (no `data:` block). | histogram_results_STAT.py + plot_results_STAT.py groomed/axis loops |
| C6 | **DyG (dynamical-grooming) variant of `ktg_alice` is never histogrammed** — the histogrammer's grooming loop skips non-soft_drop methods (`if grooming_setting.get("type") != "soft_drop": continue`, and the legacy `self.suffix` reads `z_cut`/`beta` which DyG lacks). `ktg_alice` config has both SD and DyG; the analyzer writes a DyG column but it gets no histogram. | **DONE (2026-06-08)** — the histogrammer + plotter grooming loops branch on `grooming_setting["type"]` instead of blanket-skipping: soft_drop keeps the legacy `zcut`/`beta` suffix; `dynamical_grooming` is processed for migrated observables (suffix `_R{R}_a{a}…`, guarding the `z_cut`/`beta` reads). The encoder name (`convert_to_grooming_method_spec`→`DynamicalGroomingSpec`→`…_jet_grooming_settings_a_1.0`) and the data-block key (`GroomingSettingsSpec`→`DyG_a_1.0`) already handled DyG, so no encoder change. Non-soft-drop on a NON-migrated observable is still skipped. Also fixed the ktg_alice pp DyG data-block key `jet_dynamical_grooming`→`jet_grooming_settings` (the old key vacuously matched; AA + code use `jet_grooming_settings`). Validated e2e on `Run30000_pp_small_sample`: DyG histograms produced (before/after +4 keys, 0 removed → no regression), both centralities render, pp overlay → Table 1 (SD → Table 4). `ktg_alice` is the only DyG observable. | histogram_results_STAT.py + plot_results_STAT.py (grooming loop) + config DyG key |
| C7 | **`mass_alice` / `angularity_alice` conflated an ungroomed and a groomed quantity under one observable key** — each had a legacy f-string fill for the ungroomed quantity (`jet.m()` / `λ(jet)`) AND an encoder fill for the groomed one (`m_g` / `λ(groomed pair)`), keyed per `jet_grooming_settings` (`SD_z_cut_000_beta_0` ungroomed vs `SD_z_cut_020_beta_0` groomed). | **DONE (Step 5, 2026-06-03)**: SPLIT into two observables each — `mass_alice` (ungroomed) + `mg_alice` (groomed); `angularity_alice` (ungroomed) + `angularity_groomed_alice` (groomed). Each new block declares a SINGLE `grooming_settings` entry, so `jet_grooming_settings` becomes non-essential → omitted from the encoder name (the two halves are distinguished by the observable NAME, not a grooming tag) but still selected for the per-variant HEPData table via `data_block_params`. **Both halves moved onto the encoder** (Raymond's single-path design): the ungroomed fills, previously legacy 2D `[jet_pt, value]` f-strings, are now 1D-per-pt-bin encoder fills (compute `jet_pt_spec` locally in the ungroomed method, mirror of axis_alice WTA_Standard C3/C4); the groomed fills were renamed to the new `mg_alice` / `angularity_groomed_alice`. The `_unsubtracted` QA companions were dropped (0 hists on the legacy path → no regression, as with charge_cms C8). All four added to `ENCODER_MIGRATED_JET_OBSERVABLES`; new curation-DB aliases `mg_alice` / `angularity_groomed_alice` reuse the sibling's HEPData records. See "F. Step 5 split mechanics". | ENCODER_MIGRATED_JET_OBSERVABLES (now included); analyzer ungroomed+groomed fills; config split |
| C8 | **`charge_cms` parametrized by kappa — ENCODER MIGRATION (Step 4.5, 2026-06-03)** — the analyzer wrote a legacy f-string column (`inclusive_jet_charge_cms_R0.4_k{kappa}`, 1D scalar) while the histogrammer/plotter never iterated kappa (see C9), so 0 histograms. | **migrated**: analyzer → encoder name `obs.encode_name_for_storing_in_file(jet_R, jet_charge=JetChargeSpec(kappa), tag=...)` (jet_pt is a single cut bin `[120, null]`, non-essential → omitted); `("inclusive_jet","charge_cms")` added to `ENCODER_MIGRATED_JET_OBSERVABLES`; histogrammer/plotter build the same name (`_encoder_column_name` gained a `charge=` arg, both lockstep copies) and select the per-kappa HEPData table via `data_block_params={"jet_charge": JetChargeSpec(kappa).encode()}` (kappa 0.3→Table 8, 0.5→Table 1, 0.7→Table 9). **Two intentional side-decisions:** (1) the `_unsubtracted` QA companion was **dropped** — it produced 0 histograms on the legacy path (never matched), so no regression; (2) **no `_raa_denom`** — `charge_cms`'s AA `ratio` ≡ `spectra` (same Tables 8/1/9), so `maybe_book_raa_denom` self-skips (it is a self-normalized 1/N dN/dQ distribution, not a spectrum R_AA). Verified e2e: 9 non-empty hist keys/arm (3 kappa × pop. centralities/labels), per-kappa binning resolves the right table, plotter builds matching names (9/9 hits) + per-kappa overlay, pp 18 / AA 3 charge PDFs, 0 TAxis/divide errors, Step-4 set unaffected. | analyzer charge_cms fill; histogram_results_STAT.py + plot_results_STAT.py (`_encoder_column_name`, sub-observable loop); plot_results_STAT_utils.py (`ENCODER_MIGRATED_JET_OBSERVABLES`) |
| C9 | **Sub-observable loop key-mismatch root cause** — the hist/plot sub-observable loops historically keyed on the **pre-migration** YAML names `"kappa"` and `"r"`, but the schema was renamed to `charge` (jet charge) and `subjet_R` (zr). So the parameter branch never fired → the `_k{kappa}`/`_r{r}` suffix was never built → the looked-up name didn't match the analyzer's columns → **0 histograms** for `charge_cms`, `zr_alice` (and `alpha`/`angularity_alice` had no branch at all). | **charge_cms fixed (Step 4.5)**: rename the `"kappa"` branch to read `block["jet"]["charge"]`. **`zr_alice` (`"r"` → `subjet_R`) left unfixed** — even with the loop fixed it can't bin: its `data:` block is uncurated (blank table refs, see A5). Deferred until curated; the dead `"r"` branch was removed from the histogrammer to keep the two loops in lockstep. **`angularity_alice` / `angularity_groomed_alice` (alpha) DONE (Step 5)**: a new `"angularity"` branch builds `[(f"_alpha{a['alpha']}", None, None, AngularitySpec(alpha=a["alpha"])) for a in block["jet"]["angularity"]]` (the sub-observable tuple was widened 3→4: `(label, axis_entry, charge_value, angularity_spec)`); `angularity_spec` is threaded through both lockstep `_encoder_column_name` helpers (new `angularity=` arg → `kwargs["jet_angularity"]`) and added to `data_block_params` (`jet_angularity: alpha_X_kappa_1.0`) for per-alpha table selection. **Kappa-encoding fix (latent bug):** the groomed-angularity analyzer fill passed `AngularitySpec(alpha, kappa=1)` (int) → `..._kappa_1`, but the data block keys use `kappa_1.0`; now pass `AngularitySpec(alpha)` (default kappa=1.0 float) in BOTH fills → `..._kappa_1.0` matches. (Was dormant: angularity produced 0 hists pre-Step-5.) | histogram_results_STAT.py + plot_results_STAT.py sub-observable loops |

---

## D. Histogrammer robustness (null / invalid output)

| # | Edge case | Behaviour | Where |
|---|---|---|---|
| D1 | **Degenerate binning** (`bins` is `None`, empty, or a single edge → `TH1F(…, len(bins)-1, …)` with ≤0 bins) | **protect**: `histogram_1d_observable` / `histogram_2d_observable` return early if `len(bins) < 2` | histogram_results_STAT.py:1029, 1062 |
| D2 | **ROOT `gDirectory` orphaning** — booked histograms were owned by whichever HEPData `TFile` was open (for bin lookups) and got deleted when it closed → dangling/null entries in `output_list` (165 in AA, 55 in pp before the fix). | **protect (root cause fix)**: `ROOT.TH1.AddDirectory(False)` so new histograms aren't tied to a directory; everything is written explicitly | histogram_results_STAT.py:29 |
| D3 | **Any residual null/non-writable output object** | **protect (defense-in-depth)**: `write_output_objects` skips null entries with a single summary warning instead of crashing the whole file | histogram_results_STAT.py:1154, 1168 |
| D4 | **Empty `systematics_names` silently drops the observable from `build_tables` output** — `_expand_ratio_entries` skips any entry with both `systematics_names` and `additional_systematics` empty (`Skipping unfilled ratio entry`, debug-level). This is intentional, but the failure mode is easy to misread: an unfilled block does not merely produce *zero-error* columns, it produces **no `.dat` file at all**. Found 2026-08-06 via `inclusive_chjet/ktg_alice`, which had been emitting 0 tables because its `AA.ratio` systematics were `{}` (198 → 202 tables once filled). | **by design, but scan for it**: a gap scan that only counts "unwired tables" misses this class. Count wired-table leaves whose `systematics_names` is empty — and **recurse into nested `combinations`**, since blocks like `ktg_alice` nest grooming → centrality and a single-level scan reports them as unwired. | build_tables.py:361 |

---

## Open / to-audit

- **`STAT_2760.yaml` and `STAT_200.yaml`** not yet audited for the A/B edge cases — repeat the `data:`-structure scan there.
- **A7 (no matching table for a filled MC combination)** — the full per-(observable, centrality, pt) list of skipped combinations hasn't been enumerated exhaustively; representative observables are noted. If a *physics* comparison is expected for one of these, it indicates a missing HEPData table (push to the experiment) rather than a code bug.
- **C4/C5 (encoder-name migration)** — DONE for BOTH the histogrammer (Step 4 ③, 2026-06-02) and the plotter (Step 4 ④, 2026-06-02, `47583d2`). Every migrated jet/substructure observable is now on one name path end-to-end.
- **C6 (DyG `ktg_alice`)** — DONE (2026-06-08): histogrammer + plotter loops now process dynamical_grooming for migrated observables + pp DyG data-block key fixed. Validated e2e (pp). See the C6 row above.
- **C7 (`mass_alice`/`angularity_alice` ungroomed↔groomed split)** — DONE (Step 5, 2026-06-03): split into `mass_alice`+`mg_alice` and `angularity_alice`+`angularity_groomed_alice`, all on the encoder. See "F. Step 5 split mechanics".
- **C8/C9 (`charge_cms` kappa migration + sub-observable key-mismatch)** — DONE (Step 4.5 charge; Step 5 angularity alpha). `zr_alice` (subjet_R) remains deferred (A5 — uncurated data).
- **B11 curation follow-up** — `charge_cms` Tables 1/8/9 (+ AA ratios) ship `low`/`high` SWAPPED. The plotter now normalizes (min/max) + warns, but the **source data should be fixed in the curation track** (the same swap also mis-feeds `build_tables.py`, which reads `low`/`high` raw). Track in the curation repo's `CURATION_NOTES.md`.
- **A9 for the `tgraph` overlay** — DONE (Step 4 ④): the overlay path (`tgraph_from_data_block`) now also threads `data_block_params`. `axis_alice` is the one exception (legacy ROOT overlay path) — deferred to Step 6 (see A9 caveat).
## E. Plotter render-layer fixes (pre-existing, NOT the encoder migration)

Surfaced by the first true end-to-end render after Step 4 ④ unblocked the plotter. The AA
render path in particular had never been exercised. All FIXED 2026-06-02 (the new schema
nests/renames things the never-run code paths still read the old way).

| # | Edge case | Behaviour | Where |
|---|---|---|---|
| E1 | **Acceptance cuts: schema rename `eta_cut`/`eta_cut_R`/`y_cut` → nested.** New YAML nests them under `hadron:`/`jet:` (`hadron.eta`, `jet.eta`, `jet.eta_R`, `jet.rapidity`); `init_common_settings` read only the legacy top-level keys → `self.eta_cut`/`self.y_cut` never set → `scale_histogram` `AttributeError` (hadron `pt_ch_alice`; inclusive_jet `pt_alice`/`pt_cms` for eta, `pt_atlas` for y). η (pseudorapidity) and y (rapidity) are kept SEPARATE — ATLAS jets use `rapidity`→`y_cut`, CMS jets/hadrons use `eta`→`eta_cut`, ALICE jets use `eta_R`→`eta_cut=round(eta_R−R,1)`. | **FIXED** (`f13a717`): read the nested keys (legacy kept as fallback); config-agnostic (5020/2760/200). | `plot_results_STAT.py:init_common_settings` |
| E2 | **AA-branch attribute defaults missing.** `init_common_settings`'s `is_AA` branch never defaulted `self.ytitle` (new schema comments out `ytitle_AA`) and never set `self.y_ratio_min/max` for non-v2 observables (only pp/v2 did) → `plot_RAA` `AttributeError` on `SetYTitle` / `SetRangeUser`. | **FIXED** (`51e1ffa`): default `ytitle` via `.get`, add the `y_ratio` block (per-block override else 0.0/1.99), mirroring the pp branch. | `plot_results_STAT.py:init_common_settings` (AA branch) |
| E3 | **`Data_*.dat` table write aborts on data/hist binning mismatch.** `write_experimental_data` hard-`raise`d `ValueError` when the truncated data graph didn't align bin-for-bin with the prediction histogram (e.g. `pt_cms`: HEPData doesn't cover every bin → trailing-zero x-points), aborting the whole run AFTER `plot_RAA` had already saved the R_AA PDF. | **FIXED** (`51e1ffa`): warn-and-skip the ancillary table (guard both the length mismatch and the containment check) instead of raising. The R_AA plot is unaffected. | `plot_results_STAT.py:write_experimental_data` |

**Render status:** real plotter renders pp (128 PDFs) + AA→R_AA (25 PDFs) clean, exit 0.
**⚠️ CAVEAT (open, 2026-06-02):** "renders" ≠ "validated" — the user observed the **AA R_AA
histograms look physically strange/suspect**. Not yet diagnosed; debugging deferred to the
next session (see MIGRATION_NOTES "Full render now COMPLETES … but the AA R_AA output looks
WRONG" for first suspects: AA `scale_histogram` `eta_cut` normalization, the `plot_RAA`
pp-reference division, small-sample centrality effects). E1/E2/E3 fixed the *crashes*, not
necessarily the *numbers*.
`axis_alice` is excluded from a full whole-config render — it still carries legacy
`hepdata_pp`/`hepdata_AA` keys pointing at HEPData ROOT files NOT on disk
(`data/STAT/5020/inclusive_chjet/axis_alice/HEPData-ins{2182727,2648610}-v1-root.root`), so
the legacy `tgraph_from_hepdata` overlay path `OSError`s. It is the ONLY enabled observable with
missing legacy ROOT files (xj_gamma_atlas/cms files are present). Resolve in Step 6 by stripping
its legacy `hepdata_*` keys so it routes through the (present) `data:` block.

## F. Step 5 split mechanics (mass_alice/mg_alice, angularity_alice/angularity_groomed_alice)

The two ALICE substructure observables that fused an ungroomed and a groomed quantity under one key
(C7) were split into four distinct observables, all on the encoder. Verified e2e on the pp small
sample (analyzer → histogrammer → plotter, exit 0): mass 3+3 non-empty hist keys, angularity 24+24
(4 alpha × 3 pt × {distribution + `_raa_denom`}), per-pt/per-grooming/per-alpha HEPData tables resolve
to distinct binnings + overlays, **0** "no matching table"/TAxis/divide errors, Step-4/4.5 set
unchanged.

| # | Point | Detail |
|---|---|---|
| F1 | **Single grooming per block ⇒ grooming non-essential ⇒ omitted from the name** | After the split each block has ONE `grooming_settings` entry, so `jet_grooming_settings` has `len(values)==1` → not essential → `encode_name_for_storing_in_file` excludes it. So `mass_alice` (ungroomed) and `mg_alice` (groomed) names differ ONLY by the observable name (`..._mass_alice_jet_pt_X_jet_R_0.2` vs `..._mg_alice_...`), NOT by a grooming tag. The histogrammer grooming branch still *passes* the grooming spec (popped as a benign non-essential, debug log only), so analyzer (doesn't pass it) and histogrammer (passes it) produce identical names. |
| F2 | **Table selection still uses grooming (+alpha)** | The NAME omits grooming, but the per-variant HEPData table is still selected via `data_block_params` (`jet_grooming_settings: SD_z_cut_0X0_beta_0`, plus `jet_angularity: alpha_X_kappa_1.0` for angularity). `jet_pt` is matched from the `_pt{i}` suffix (the binning lookup keeps `pt_suffix`; only the hist NAME suppresses it). So mass_alice vs mg_alice and each alpha resolve to their own table/binning. |
| F3 | **Ungroomed half migrated to the encoder (not kept legacy)** | Per Raymond's single-path design, the ungroomed fills moved from legacy 2D `[jet_pt, value]` f-strings to 1D-per-pt-bin encoder fills. They live in the *ungroomed* analyzer method (no grooming loop), so `jet_pt_spec` is computed locally there (`next(PtSpec(lo,hi) for ... if lo <= jet_pt < hi)`), exactly like the axis_alice WTA_Standard migration (C3/C4). |
| F4 | **`_unsubtracted` companions dropped** | The shower_recoil `_unsubtracted` QA columns were removed (they were 0-hist on the legacy path → no regression, same decision as charge_cms C8). The hole-subtraction physics is unchanged. |
| F5 | **Kappa float vs int (latent bug fixed)** | `AngularitySpec(alpha, kappa=1)` (int) encodes `..._kappa_1`; the data block keys are `kappa_1.0`. Both fills now pass `AngularitySpec(alpha)` (default `kappa=1.0` float) → `..._kappa_1.0` matches. Dormant pre-Step-5 (angularity was 0 hists). |
| F6 | **`_raa_denom` booked for angularity but not mass** | `maybe_book_raa_denom` books a pp histogram on the AA-`ratio` binning only when it differs from the pp-`spectra` binning. mass pp+AA are the SAME record (ins2845788) → same binning → skipped (3 keys/arm). angularity pp (ins1891385) and AA (ins2845788) are DIFFERENT records → different binning → `_raa_denom` booked (so 24 = 12 distribution + 12 denom). Correct per-observable behavior; the denom is excluded from `keys_to_plot`. |
| F7 | **`self_normalize` matches angularity coincidentally via the `"g"` substring** | `plot_jet_observables` sets `self_normalize` if any of `["mass","g","ptd","charge","mg","zg","tg","ktg","xj"]` is a substring of the observable name. `"g"` ∈ `"angularity_alice"` → `self_normalize=True` (correct: these are `1/σ dσ/dλ` self-normalized distributions). It WORKS but is fragile (single-letter match); a config-driven `self_normalize` flag would be cleaner — latent cleanup, not a Step-5 blocker. |
| F8 | **Curation-DB aliases** | `5020/inclusive_chjet/mg_alice` and `.../angularity_groomed_alice` were added to `hepdata_database.yaml` (the curation repo), each `directory:` pointing at the EXISTING sibling record (mass→ins2845788; angularity_groomed→ins1891385 pp + ins2845788 AA) — no new HEPData files. Logged in the curation repo's `CURATION_NOTES.md`. |
| F9 | **AA arm pp-verified only** | All four are `centrality: [[0,10]]`; the PbPb small sample is ~10–11% → AA MC is empty for them (Step 7 centrality mechanics). The AA `ratio` binning/overlay code path resolves (curated), but full AA validation needs a 0–10% sample (Step 7). |

## G. Semi-inclusive recoil (Δ_recoil: IAA_pt_alice / dphi_alice) + zr_alice — Step 6 curation (2026-06-04)

| # | Edge case | Behaviour |
|---|---|---|
| G1 | **Δ_recoil is assembled, not analyzed, and MUST be assembled POST-AGGREGATION** — the analyzer produces only per-trigger-class yields (`_lowTrigger`/`_highTrigger`) + the trigger-pt (N_trig) histogram; the physical observable Δ_recoil = (1/N_trig^high)·high − c_ref·(1/N_trig^low)·low is built in the **plotter** (`construct_semi_inclusive_histogram`, routed from `get_histogram` for `observable_type=="hadron_trigger_chjet"`), reading the (hadd-merged) raw high/low/N_trig from the input file. N_trig via `FindBin` at the low/high range midpoints; dphi reuses the IAA-owned trigger-pt histogram (shared trigger). **It must NOT be built per-file in the histogrammer**: the 1/N_trig normalization is non-linear, so per-file construction + `hadd` over-counts by **n_files** (the 30-file sample showed IAA ×30 high; fixed 2026-06-04 by moving the build to the plotter). | **post-aggregation**: histogrammer books only raw high/low/N_trig; plotter builds Δ_recoil after the merge. `plot_hadron_trigger_chjet_observables` (existed but was **never invoked**) is wired into `plot_results()`, with a pt-bin loop for dphi. |
| G2 | **Per-trigger normalization must NOT get the standard scaling** — Δ_recoil is per-*trigger*, not per-event. The 1/N_trig ratio cancels the `xsec/weight_sum` factor every other observable needs, so `_scale_one_histogram` early-returns for `hadron_trigger_chjet` (else ~1e3 double-normalization + a second width). **Guard:** `and not self_normalize` so the self-normalized `nsubjettiness_alice` (STAT_2760) still self-normalizes. pp uses c_ref=1; the AA-arm c_ref list is applied only on `is_AA`. | **protect**: skip standard chain; keep self-normalize for nsubjettiness. |
| G3 | **`min_jet_pt` floor truncates the lowest recoil-jet-pt bins** — jet finding applies a global `fj.SelectorPtMin(min_jet_pt)` (config top-level), so no jets below it exist for ANY observable. IAA/dphi are the only enabled observables reaching below 20 GeV (IAA data → 7 GeV; dphi pt0 = 10-20). With `min_jet_pt: 20` those bins were **empty** (not a bug). Lowered to **7** (2026-06-04); **needs the slow re-analysis** to take effect. Other observables cut higher, so unaffected. | **sample setting**: lower `min_jet_pt` + re-analyze to fill low bins. |
| G4 | **dphi Δ_recoil needs a recoil-jet-pt-window division (RESOLVED at full stats 2026-06-04)** — the HEPData `dphi` Δ_recoil(Δφ) is **double-differential**, per rad AND per recoil-jet-pt `(GeV/c·rad)^-1`. The MC histogram is binned in Δφ but **integrated over the jet-pt sub-bin window**, so it is high by that window width (e.g. ×10 for the 20-30 GeV bin). The small sample masked it (ratio≈1 within the wide band); the 30-file sample showed it clearly. **Fix:** in `construct_semi_inclusive_histogram`, divide the dphi Δ_recoil by `jet.pt[i+1]-jet.pt[i]`. After the fix, dphi ratio ≈ 1. **IAA needs NO analogous factor** — its pt-spectrum data is Δφ-integrated (per GeV, not per GeV·rad), and it already matches at ratio ≈ 1 (so the review's suggested IAA /1.2 was NOT applied — it would have made a good agreement worse). | **protect**: dphi /pt-window; IAA unchanged. |
| G5 | **zr_alice subjet_R — legacy-path sub-observable migration (C9 resolved)** — added a `subjet_R` branch to the jet sub-observable loop (tuple widened 4→5) in both histogrammer + plotter, emitting the `_r{r}` legacy suffix (matches the analyzer column `..._R{R}_r{r}`; zr stays OFF the encoder) and threading `data_block_params={jet_subjet_R: r_X}` (NOT gated on `is_migrated`) for the per-r binning + overlay. `"zr"` added to the self-normalize trigger (z_r is 1/σ self-normalized). Curated: Tables 5/8 (pp), 6/9 (AA), 7/10 (PbPb/pp ratio). **Review fix:** the legacy `maybe_book_raa_denom` call omitted `data_block_params`, so the r=0.2 R_AA denom silently resolved to the r=0.1 ratio table → now forwarded (no-op for non-subjet observables, where it is `None`). | **DONE** pp-validated (ratio≈1). |
| G6 | **AA R_AA crash on hist/graph x-mismatch (FIXED 2026-06-04)** — `truncate_tgraph` aligns the MC histogram bins to the HEPData graph points; when the MC histogram extends past the data x-range (graph x reads 0 once exhausted, e.g. `inclusive_jet/pt_cms` at high pt), it found a mismatch. For the **AA arm it RAISED**, aborting the *entire* R_AA render on the first such observable (only ~24 of 72 R_AA PDFs were produced before the crash). Surfaced by the Step-7 large-sample R_AA run (10-40% centrality). | **protect**: warn + skip that observable's overlay/ratio gracefully (`return None`) for AA too, matching the pp behaviour — the rest of the R_AA plots now complete (72 PDFs). `plot_results_STAT_utils.py:truncate_tgraph`. |

---

## H. STAT_5020 curation completeness pass (2026-08-06)

Goal: make the 5020 config as complete as the available HEPData allows. Three
findings, all curation-side (no framework code changed).

| # | Item | Detail |
|---|---|---|
| H1 | **`systematics_names` was filled only on `AA.ratio` for the April-2026 RAA observables** | 13 wired blocks across `hadron/{pt_ch_alice,pt_pi_alice,pt_ch_cms,pt_ch_atlas}` and `inclusive_jet/{pt_alice,pt_atlas,pt_y_atlas}` had their `pp.spectra` / `AA.spectra` tables wired but `systematics_names: {}`, so every pp and PbPb *spectrum* overlay drew with zero systematic errors. Now filled; all label sets verified uniform across every bin and column, and two cross-checked against raw HEPData (`pt_ch_alice` pp bin 0 → 0.2281; `pt_atlas` pp bin 0 → √(22.24²+0.22²+2.56²) = 22.39, i.e. asymmetric errors + the separate luminosity term both propagate). 2760 and 200 already followed the fuller spectra+ratio+pp convention — 5020 was the outlier. |
| H2 | **`pt_pi_alice` publishes `syst.` AND `syst. uncorr.` — the latter is a SUBSET** | `syst. uncorr.` is always smaller than `syst.` (e.g. 39.9 vs 146.0): it is the bin-uncorrelated *portion* of the total, not an independent component, so mapping both would double-count in quadrature. Only the total is mapped, matching this observable's own `AA.ratio` block. **If the decomposition is ever wanted for covariance work**, the correct correlated part is `sqrt(syst² − uncorr²)`, which `build_tables` cannot express today — it would need a derived-column mechanism. |
| H3 | **`ktg_alice` `AA.spectra` had Soft Drop and Dynamical Grooming SWAPPED** | The HEPData descriptions state the grooming explicitly: Figure 1 **right** = Soft Drop (pp Table 4, PbPb 0-10 Table 6, 30-50 Table 5); Figure 1 **left** = Dynamical Grooming (pp Table 1, PbPb 0-10 Table 3, 30-50 Table 2). The config assigned SD→Tables 3/2 and DyG→Tables 6/5, i.e. each grooming overlaid the *other* grooming's PbPb distribution. `pp.spectra` and `AA.ratio` were correct, so only the AA distribution overlay was affected. Fixed; all 10 grooming×centrality×artifact mappings now verify against the record. Its AA systematics were also empty — see **D4**, which is why it had been emitting zero `.dat` tables. |

**New observable enabled: `inclusive_jet/rg_atlas`** (ATLAS soft-drop r_g R_AA,
ins2512925 / arXiv:2211.11470). HEPData was published after the block was first
stubbed out, so the config still said `hepdata: N/A`, `enabled: false`. The
analyzer already supported it (`ENCODER_MIGRATED_JET_OBSERVABLES`), so this was
purely a data-side task. Wired `pp.spectra` → Table 2 and `AA.ratio` → Tables
15-18 (4 centralities × 3 pt bins), and updated the selections to the published
binning — `centrality` `[[0,10]]` → `[[0,10],[10,30],[30,50],[50,80]]`, `jet.pt`
`[100,150,200]` → `[158,200,315,501]` — plus removed the now-wrong `skip_pp` and
the stub `bins: [0.0, 1.0]`. See **A11** (why `AA.spectra` is deliberately blank)
and **A12** (the `hepdata_database.yaml` registration step). Validated: pp/AA bins
identical (12 bins, so the R_AA divide aligns), R_AA rises monotonically with
centrality (0-10% ≈ 0.71-0.84 → 50-80% ≈ 0.91-1.00), analyzer and histogrammer
encoder names agree byte-for-byte, +12 `.dat` tables.

**FLAGGED, not changed — `rg_atlas` acceptance.** Every other ATLAS inclusive-jet
observable declares acceptance as `rapidity` (`Dz_atlas` 2.1, `pt_atlas` /
`pt_y_atlas` 2.8); `rg_atlas` alone uses `eta: 2.0`, which the analyzer applies as
`|eta| < eta - R` (= 1.6). The HEPData record does not state the acceptance, so
confirming the right value needs the paper. Left as-is pending that check.

### H4. Unwired-slot audit (2026-08-06) — 112 of 121 are genuine HEPData gaps

Every unwired slot in the enabled 5020 config was checked against its record.
**Read every dependent-variable column, not just the first** — `mg_cms`'s first
columns all say `CENTRALITY: 0-10`, but dv3-6 of Figure 3a carry 10-30/30-50.

Genuine gaps (nothing to curate): `axis_alice` 45, `zg_cms` 30, `mg_cms` 24,
`tg_alice` 6, `zg_alice` 3, `axis_cms` 1 (no pp table at all), `mass_alice`/
`mg_alice` pp 2 (pp jet mass published only to 100 GeV), `rg_atlas` 1 (A11).
Both CMS substructure records are a **"cross"** in the (centrality × pt) plane —
the pt dependence published only at 0-10%, the centrality dependence only in one
pt bin — and the config wiring already traces that cross exactly.

Wired this session: `hadron/pt_ch_atlas` AA.spectra → Tables 11-16, and
`inclusive_chjet/pt_alice` AA.spectra → "Figure 3a/3b top R0XX" (the June
straggler). Left as-is deliberately: `mass_alice`, `mg_alice`, `angularity_alice`,
`angularity_groomed_alice` AA.spectra — under the spectra-as-ratio convention their
`AA.ratio` already points at the PbPb spectra tables, and with `skip_AA_ratio` the
resolver falls back spectra→ratio, so wiring them would only duplicate references.

**`zg_alice` {0-10%, R=0.4} is NOT a curation task.** The measurement exists
(Tables 13-15) but at pt **80-100**, while the config has a single 60-80 bin.
Recovering it is a multi-file behaviour change, not a table reference:
1. `analyze_events_STAT.py` hardcodes a single window (`pt[0]`/`pt[1]`) and calls
   `encode_name_for_storing_in_file(**_parameters, tag=...)` **without** `jet_pt`.
   A second pt bin makes `jet_pt` essential → that call raises `KeyError`. It needs
   the per-bin `PtSpec` lookup used by `axis_alice`/`rg_atlas` directly below it.
2. `tg_alice` is filled in the SAME `if` block, gated entirely on **zg_alice's**
   config, so widening zg_alice's `pt` also re-bins tg_alice — which has no data at
   80-100 (theta_g R=0.4 / 0-10% is unpublished at any pt).
3. `histogram_results_STAT.py:761` hardcodes a skip dropping exactly
   (R=0.4, cent 0-10) and (R=0.2, cent 30-50) — i.e. the code already encodes this
   HEPData gap — and it is SHARED by zg_alice and tg_alice. It would have to become
   pt-bin-aware for zg_alice only.
4. Column names change for both observables → existing analyzer output invalid,
   full re-analysis required, for **one** extra data/MC comparison point.
