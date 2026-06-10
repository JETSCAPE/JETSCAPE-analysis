"""Utils for histogramming / plotting

.. codeauthor:: James Mulligan <james.mulligan@berkeley.edu>, UC Berkeley
"""

from __future__ import annotations

import ctypes
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import ROOT  # pyright: ignore [reportMissingImports]
import yaml
from jetscape_analysis.base import common_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

logger = logging.getLogger(__name__)

# One-shot guards so recurring data-quality warnings (e.g. swapped HEPData bin edges, seen on
# every bin-read) are logged once per process rather than once per call.
_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned_once:
        _warned_once.add(key)
        logger.warning(message)

# Jet/substructure observables whose ANALYZER fill is migrated to the observable encoder
# (obs.encode_name_for_storing_in_file). For these, the histogrammer and plotter must derive the
# storage column / histogram name from the same encoder rather than hand-building a legacy
# f-string, otherwise the lookup misses and the observable silently produces no histogram (Step 4;
# OBSERVABLE_EDGE_CASES C5). Keyed (observable_type, observable). Everything not listed here
# (e.g. charge_cms, plain RAA/Dz/...) stays on the legacy f-string path, matching its analyzer.
# Lives here (shared utils) so the histogrammer and plotter use a single source of truth and can
# never drift on membership.
# NOTE: Step 5 split the two ALICE observables that conflated an ungroomed quantity with a groomed one
# (each previously had a legacy f-string fill for jet.m()/lambda(jet) AND an encoder fill for
# m_g/lambda(groomed)). mass_alice -> mass_alice (ungroomed) + mg_alice (groomed); angularity_alice ->
# angularity_alice (ungroomed) + angularity_groomed_alice (groomed, parametrized by the alpha
# sub-observable loop). All four are now fully on the encoder, listed below.
ENCODER_MIGRATED_JET_OBSERVABLES = {
    ("inclusive_chjet", "ktg_alice"),
    ("inclusive_chjet", "zg_alice"),
    ("inclusive_chjet", "tg_alice"),
    ("inclusive_chjet", "axis_alice"),
    ("inclusive_jet", "mg_cms"),
    ("inclusive_jet", "zg_cms"),
    ("inclusive_jet", "rg_atlas"),
    ("inclusive_jet", "axis_cms"),
    # Step 4.5: jet charge, parametrized by kappa (data block keyed by jet_charge: kappa_X).
    ("inclusive_jet", "charge_cms"),
    # Step 5: ALICE jet mass, split into ungroomed (mass_alice, jet.m()) + groomed (mg_alice, m_g).
    ("inclusive_chjet", "mass_alice"),
    ("inclusive_chjet", "mg_alice"),
    # Step 5: ALICE jet angularity, split into ungroomed + groomed, each parametrized by alpha
    # (data block keyed by jet_angularity: alpha_X_kappa_1.0).
    ("inclusive_chjet", "angularity_alice"),
    ("inclusive_chjet", "angularity_groomed_alice"),
}


def available_hepdata_files_in_block(block: dict[str, Any]) -> list[str]:
    """Determine the hepdata files available in the config block.

    Return:
        List of hepdata files available in the block.
    """

    available_hepdata_files = [v for v in ["hepdata", "hepdata_pp", "hepdata_AA"] if v in block]
    # Expect to have either hepdata or if in separate files, it could be hepdata_pp and hepdata_AA
    # Validate this:
    if available_hepdata_files and "hepdata_pp" in available_hepdata_files:
        assert "hepdata_AA" in available_hepdata_files, "If hepdata_pp is present, hepdata_AA must also be present."

    return available_hepdata_files


################################################################
class PlotUtils(common_base.CommonBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ---------------------------------------------------------------
    # Get bin array specified in config block
    # ---------------------------------------------------------------
    def bins_from_config(
        self, block, sqrts, observable_type, observable, centrality, centrality_index, suffix="", is_AA=False, pt_suffix="",
        data_block_params=None,
    ) -> npt.NDarray[np.int32] | npt.NDArray[np.int64]:
        if "bins" in block:
            logger.info(f"  Histogram with custom binning found for {observable} {centrality} {suffix}")
            return np.array(block["bins"])
        if "hepdata" in block:
            logger.info(f"  Histogram with hepdata binning found for {observable} {centrality} {suffix}")
            return self.bins_from_hepdata(block, sqrts, observable_type, observable, centrality_index, suffix)
        if "data" in block:
            logger.info(f"  Histogram with data-block (HEPData YAML) binning found for {observable} {centrality} {suffix}")
            # is_AA selects the measured artifact (AA->ratio, pp->spectra), which
            # determines the binning the histogram must match.
            return self.bins_from_data_block(
                block, is_AA, sqrts, observable_type, observable, centrality, suffix, pt_suffix, data_block_params
            )
        logger.warning(f"\tNo binning found for {observable} {centrality} {suffix}")
        return np.array([])

    # ---------------------------------------------------------------
    # Get bin array from hepdata file specified in config block
    # ---------------------------------------------------------------
    def bins_from_hepdata(self, block, sqrts, observable_type, observable, centrality_index, suffix=""):
        # Open the HEPData file
        hepdata_dir = Path(f"data/STAT/{sqrts}/{observable_type}/{observable}")
        hepdata_filename = hepdata_dir / block["hepdata"]
        f = ROOT.TFile(str(hepdata_filename), "READ")

        # Find the relevant directory:
        # - We require that the centrality index is specified -- but
        #   the directory name may be found in the config either
        #   as a list of dirs, or a single dir with a list of histogram names
        # - The list of dir/hist names may also contain a suffix,
        #   which specifies e.g. the pt bin, jetR, or other parameters

        # First, check for dir names in config
        if f"hepdata_AA_dir{suffix}" in block:
            dir_key = f"hepdata_AA_dir{suffix}"
        elif "hepdata_AA_dir" in block:
            dir_key = "hepdata_AA_dir"
        else:
            logger.info(f"hepdata_AA_dir{suffix} not found!")
            return np.array([])

        # Check for hist names in config
        if f"hepdata_AA_hname{suffix}" in block:
            h_key = f"hepdata_AA_hname{suffix}"
        elif "hepdata_AA_hname" in block:
            h_key = "hepdata_AA_hname"
        else:
            logger.info(f"hepdata_AA_hname{suffix} not found!")
            return np.array([])

        # Get the appropriate centrality entry in the dir/hist list
        if type(block[dir_key]) is list:
            # If fewer entries than the observable's centrality, skip
            if centrality_index > len(block[dir_key]) - 1:
                return np.array([])

            dir_name = block[dir_key][centrality_index]
            h_name = block[h_key]

        elif type(block[h_key]) is list:
            # If fewer entries than the observable's centrality, skip
            if centrality_index > len(block[h_key]) - 1:
                return np.array([])

            dir_name = block[dir_key]
            h_name = block[h_key][centrality_index]

        else:
            # Both are plain strings
            dir_name = block[dir_key]
            h_name = block[h_key]

        # Get the histogram, and return the bins
        dir = f.Get(dir_name)
        h = dir.Get(h_name)
        bins = np.array(h.GetXaxis().GetXbins())

        # For certain Soft Drop observables, we need to exclude the "untagged" bin so that it will become underflow
        if observable in ["zg_alice", "tg_alice"]:
            bins = bins[1:]
        # For ATLAS RAA y-dependence, add a bin at 0<abs(y)<0.3 (it is not included in the hepdata since they take a ratio to that bin)
        if observable == "pt_y_atlas":
            bins = np.insert(bins, 0, 0.0, axis=0)

        f.Close()

        return bins

    # ---------------------------------------------------------------
    # Turn a HEPData independent-variable `values` list into a binning.
    # Returns (edges, centers, exl, exh) as float64 arrays, or None if unusable.
    # Handles both {low, high} per-bin tables and center-only {value} tables; for
    # the latter, edges are synthesized as midpoints and `centers` are the bin
    # centers (NOT the raw values), so a histogram binned with `edges` has bin
    # centers that coincide with the graph x-points — keeping the two aligned.
    # ---------------------------------------------------------------
    @staticmethod
    def _axis_from_independent_values(values):
        if not values or not all(isinstance(v, dict) for v in values):
            return None
        if all("low" in v and "high" in v for v in values):
            lo = np.array([float(v["low"]) for v in values], dtype=np.float64)
            hi = np.array([float(v["high"]) for v in values], dtype=np.float64)
            # Some curated HEPData tables store the bin boundaries with low/high swapped
            # (low > high per bin -- e.g. charge_cms Tables 1/8/9 and their AA ratio tables).
            # Normalize lower/upper = min/max so each bin is well-ordered; a well-formed
            # (low < high) table is unaffected. Warn rather than silently absorb, so the
            # underlying data bug stays visible for curation.
            if np.any(lo > hi):
                _warn_once(
                    "swapped_bins",
                    "\tHEPData bin has low > high (swapped); normalizing via min/max -- flag for curation.",
                )
            lows = np.minimum(lo, hi)
            highs = np.maximum(lo, hi)
            edges = np.append(lows, highs[-1])
            # The synthesized edges are monotonic only if the rows are in ascending bin order
            # (the min/max above fixes a per-bin swap, not reversed/overlapping rows). Guard
            # rather than hand a non-increasing array to ROOT TH1F (which errors + books a
            # broken axis); None is handled by callers (bins_from_data_block -> empty -> skip).
            if not np.all(np.diff(edges) > 0):
                logger.warning("\tHEPData independent-variable edges are not monotonically increasing; skipping binning.")
                return None
            centers = 0.5 * (lows + highs)
            return edges, centers, centers - lows, highs - centers
        if all("value" in v for v in values):
            raw = np.array([float(v["value"]) for v in values], dtype=np.float64)
            # Can't infer a bin width from a single center, and non-increasing
            # centers would synthesize non-monotonic edges that break ROOT TH1F.
            if raw.size < 2 or not np.all(np.diff(raw) > 0):
                return None
            mids = 0.5 * (raw[1:] + raw[:-1])
            first = 2 * raw[0] - mids[0]
            last = 2 * raw[-1] - mids[-1]
            edges = np.concatenate(([first], mids, [last]))
            centers = 0.5 * (edges[1:] + edges[:-1])
            return edges, centers, centers - edges[:-1], edges[1:] - centers
        return None

    # ---------------------------------------------------------------
    # Resolve the HEPData YAML table for a new-schema `data:` block.
    #
    # Raymond's migration replaced the old top-level `hepdata:` ROOT-file lookup
    # with a `data:` block that points at the hard-sector-data-curation HEPData
    # YAML files. This shared resolver (used by both bins_from_data_block and
    # tgraph_from_data_block) selects the table matching the requested
    # (centrality, jet_R, jet_pt) and returns the parsed per-table YAML dict plus
    # the matched expanded config entry (which carries table, index, parameters,
    # additional_systematics). Returns (None, None) on any failure.
    #
    # Artifact selection mirrors the measured quantity being compared against,
    # matching the old hepdata_{AA,pp}_dir keys:
    #   - AA -> the `ratio` block (the measured R_AA)
    #   - pp -> the `spectra` block (the measured pp spectrum)
    # with a fallback to the other artifact when the preferred one is absent.
    # This matters for binning too: e.g. pt_ch_alice R_AA is 39 bins (ratio)
    # vs 61 bins (AA spectrum) — the histogram must match the measurement.
    # ---------------------------------------------------------------
    def _resolve_data_hepdata_table(
        self, block, is_AA, sqrts, observable_type, observable, centrality, suffix="", pt_suffix="", data_block_params=None
    ):
        # Lazy imports: keep ROOT-only callers free of the data_curation deps.
        from jetscape_analysis.data_curation import data as data_mod  # noqa: PLC0415
        from jetscape_analysis.data_curation import hepdata_utils  # noqa: PLC0415

        # Read (and cache) the HEPData record database. We parse it with pyyaml
        # directly rather than hepdata_utils.read_database(), which depends on
        # ruamel.yaml — not installed in the analysis container. BASE_DATA_DIR
        # auto-falls back to the sibling clone when the submodule isn't initialized.
        if getattr(self, "_hepdata_database", None) is None:
            db_path = hepdata_utils.BASE_DATA_DIR / hepdata_utils.DEFAULT_DATABASE_NAME
            with open(db_path) as f:
                self._hepdata_database = yaml.safe_load(f) or {}
        infos = self._hepdata_database.get(f"{sqrts}/{observable_type}/{observable}")
        if not infos:
            logger.warning(f"\tNo HEPData database entry for {observable}; cannot resolve data table")
            return None, None

        system = "AA" if is_AA else "pp"
        hepdata = block.get("data", {}).get(system, {}).get("hepdata", {})
        # AA normally overlays the measured ratio (R_AA); pp overlays the spectrum. A self-normalized
        # DISTRIBUTION observable (skip_AA_ratio) overlays the measured PbPb distribution, NOT the ratio
        # -- so prefer `spectra` on the AA arm when skip_AA_ratio is set. Fall back to the OTHER artifact
        # when the preferred one has no WIRED tables: curators stored some distributions in `ratio`
        # (mass/angularity/mg) and others in `spectra` (axis_alice/charge_cms). A present-but-empty block
        # is truthy, so a plain `if not artifact` fallback would silently drop the overlay.
        def _has_wired_tables(art):
            return bool(art) and any(
                c.get("table") for t in art.get("tables", []) for c in (t.get("combinations") or [t])
            )

        prefer_spectra = (not is_AA) or block.get("skip_AA_ratio", False)
        first, second = ("spectra", "ratio") if prefer_spectra else ("ratio", "spectra")
        artifact = hepdata.get(first)
        if not _has_wired_tables(artifact) and _has_wired_tables(hepdata.get(second)):
            artifact = hepdata.get(second)
        if not artifact or "tables" not in artifact:
            logger.warning(f"\tNo data artifact (spectra/ratio) for {observable} {system}")
            return None, None

        # Expand the `tables` (with their nested `combinations`) into one flat
        # config per (table, parameters) — same helper data_curation uses.
        expanded = data_mod.expand_parameter_combinations_into_individual_configs(artifact["tables"])

        # Build the desired parameter selection. centrality arrives as a
        # [low, high] list -> the YAML uses "low_high". jet_R and jet_pt are
        # encoded in the suffix/pt_suffix (e.g. "_R0.4", "_pt2").
        combined = f"{suffix or ''}{pt_suffix or ''}"
        desired = {}
        if isinstance(centrality, (list, tuple)) and len(centrality) == 2:
            desired["centrality"] = f"{int(centrality[0])}_{int(centrality[1])}"
        m = re.search(r"_R([0-9.]+)", combined)
        if m:
            desired["jet_R"] = f"{float(m.group(1)):g}"
        mp = re.search(r"_pt(\d+)", combined)
        if mp and isinstance(block.get("jet", {}).get("pt"), list):
            i = int(mp.group(1))
            pt_edges = block["jet"]["pt"]
            if i + 1 < len(pt_edges):
                desired["jet_pt"] = f"{pt_edges[i]}_{pt_edges[i + 1]}"

        # Sub-observable parameters (jet_grooming_settings, jet_axis, ...) the caller has already
        # encoded in the data-block convention (GroomingSettingsSpec/JetAxisDifferenceSpec encoding,
        # which carries the "SD_"/"DyG_" prefix -- distinct from the analyzer's storage encoding).
        # Without these, observables with one HEPData table per grooming/axis variant would all match
        # the first table (centrality/jet_R/jet_pt only) and so share the wrong bin edges. Entries
        # that don't constrain a key still impose no condition (see _matches).
        if data_block_params:
            desired.update(data_block_params)

        def _matches(entry_params):
            # An entry matches if every parameter it constrains that we also
            # know about agrees. Entries that don't constrain a key impose no
            # condition on it.
            for k, v in entry_params.items():
                if k in desired:
                    a = f"{float(v):g}" if k == "jet_R" else str(v)
                    if a != desired[k]:
                        return False
            return True

        entry = next((e for e in expanded if e.get("table") and _matches(e.get("parameters", {}))), None)
        if entry is None:
            logger.warning(f"\tNo matching data table for {observable} {centrality} {combined} (desired={desired})")
            return None, None

        # Resolve the HEPData record for this table. Prefer the artifact's own `record`, falling
        # back to the parent `hepdata.record` (the config commonly sets the inspire id only at the
        # parent level). When the id is unresolved, a silent infos[0] fallback previously overlaid
        # the WRONG record -- e.g. the pp distribution on the PbPb canvas, since the same table name
        # ("Table 2", ...) exists in both the pp and PbPb records -- so warn loudly instead.
        record_id = artifact.get("record", {}).get("inspire_hep_id") or hepdata.get("record", {}).get("inspire_hep_id")
        info = next((i for i in infos if i.get("inspire_hep_id") == record_id), None)
        if info is None:
            if record_id is not None:
                logger.warning(
                    f"\tHEPData record ins{record_id} not among the {len(infos)} candidate(s) for "
                    f"{observable} {system}; falling back to ins{infos[0].get('inspire_hep_id')}"
                )
            elif len(infos) > 1:
                logger.warning(
                    f"\tNo HEPData record id resolved for {observable} {system} with {len(infos)} "
                    f"candidate records; ambiguously using ins{infos[0].get('inspire_hep_id')}"
                )
            info = infos[0]
        rel_filename = info.get("tables_to_filenames", {}).get(entry["table"])
        if rel_filename is None:
            logger.warning(f"\t'{entry['table']}' not in HEPData record ins{info.get('inspire_hep_id')} for {observable}")
            return None, None
        yaml_path = hepdata_utils.BASE_DATA_DIR / "data" / info["directory"] / rel_filename
        if not yaml_path.exists():
            logger.warning(f"\tHEPData YAML missing: {yaml_path}")
            return None, None
        with yaml_path.open() as f:
            hd = yaml.safe_load(f)
        return hd, entry

    # ---------------------------------------------------------------
    # Get bin array from the new-schema `data:` block (HEPData YAML).
    # Reads the bin edges from independent_variables[0] of the table that the
    # measured quantity uses (AA->ratio, pp->spectra; see _resolve_data_hepdata_table).
    # ---------------------------------------------------------------
    def bins_from_data_block(
        self, block, is_AA, sqrts, observable_type, observable, centrality, suffix="", pt_suffix="", data_block_params=None
    ):
        hd, _entry = self._resolve_data_hepdata_table(
            block, is_AA, sqrts, observable_type, observable, centrality, suffix, pt_suffix, data_block_params
        )
        if hd is None:
            return np.array([])

        # Bin edges come from the first independent variable.
        indep = (hd or {}).get("independent_variables") or []
        axis = self._axis_from_independent_values(indep[0].get("values") if indep else None)
        if axis is None:
            logger.warning(f"\t{observable}: cannot determine bin edges from independent variable")
            return np.array([])
        bins = axis[0]

        # Preserve the special cases from the old ROOT path:
        # - zg/tg_alice: drop the leading "untagged" bin so it becomes underflow.
        if observable in ["zg_alice", "tg_alice"]:
            bins = bins[1:]
        # - pt_y_atlas: HEPData omits the 0<|y|<0.3 bin (it's the ratio denominator);
        #   re-insert it.
        if observable == "pt_y_atlas":
            bins = np.insert(bins, 0, 0.0, axis=0)

        return bins

    # ---------------------------------------------------------------
    # Get tgraph from hepdata file specified in config block
    # ---------------------------------------------------------------
    def tgraph_from_hepdata(  # noqa: C901
        self, block, is_AA, sqrts, observable_type, observable, centrality_index, suffix="", pt_suffix=""
    ):
        # Open the HEPData file
        hepdata_file_key = "hepdata"
        hepdata_files_in_block = available_hepdata_files_in_block(block)
        if len(hepdata_files_in_block) > 1:
            hepdata_file_key = "hepdata_AA" if is_AA else "hepdata_pp"
        hepdata_dir = Path(f"data/STAT/{sqrts}/{observable_type}/{observable}")
        hepdata_filename = hepdata_dir / block[hepdata_file_key]
        f = ROOT.TFile(str(hepdata_filename), "READ")

        # Find the relevant directory:
        # - The list of dir/hist names may contain a suffix,
        #   which specifies e.g. the pt bin, jetR, or other parameters

        system = "AA" if is_AA else "pp"

        # First, check for dir names in config
        if f"hepdata_{system}_dir{suffix}" in block:
            dir_key = f"hepdata_{system}_dir{suffix}"
        elif f"hepdata_{system}_dir{suffix}{pt_suffix}" in block:
            dir_key = f"hepdata_{system}_dir{suffix}{pt_suffix}"
        elif f"hepdata_{system}_dir" in block:
            dir_key = f"hepdata_{system}_dir"
        else:
            # logger.info(f'hepdata_pp_dir{suffix} not found!')
            return None

        # Check for hist names in config
        if f"hepdata_{system}_gname{suffix}" in block:
            g_key = f"hepdata_{system}_gname{suffix}"
        elif f"hepdata_{system}_gname{suffix}{pt_suffix}" in block:
            g_key = f"hepdata_{system}_gname{suffix}{pt_suffix}"
        elif f"hepdata_{system}_gname" in block:
            g_key = f"hepdata_{system}_gname"
        else:
            # logger.info(f'hepdata_{system}_gname{suffix} not found!')
            return None

        # Get the appropriate centrality entry in the dir/hist list
        if type(block[dir_key]) is list:
            # If fewer entries than the observable's centrality, skip
            if centrality_index > len(block[dir_key]) - 1:
                return np.array([])

            dir_name = block[dir_key][centrality_index]
            g_name = block[g_key]

        elif type(block[g_key]) is list:
            # If fewer entries than the observable's centrality, skip
            if centrality_index > len(block[g_key]) - 1:
                return np.array([])

            dir_name = block[dir_key]
            g_name = block[g_key][centrality_index]

        else:
            dir_name = block[dir_key]
            g_name = block[g_key]

        # Get the tgraph, and return the bins
        dir = f.Get(dir_name)
        try:
            g = dir.Get(g_name)
        except TypeError:
            logger.warning(f"HEPData directory '{dir_name}' not found in {hepdata_filename}")
            f.Close()
            return None
        g = dir.Get(g_name)
        if not g:
            logger.warning(f"TGraph '{g_name}' not found in directory '{dir_name}' in {hepdata_filename}")
            f.Close()
            return None
        f.Close()

        return g

    # ---------------------------------------------------------------
    # Get tgraph from data points specified in config block
    # ---------------------------------------------------------------
    def tgraph_from_yaml(
        self, block, is_AA, sqrts, observable_type, observable, centrality_index, suffix="", pt_suffix=""
    ):
        # Load yaml file containing the data
        data_dir = Path(f"data/STAT/{sqrts}/{observable_type}/{observable}")
        data_filename = data_dir / block["custom_data"]
        with data_filename.open() as stream:
            data = yaml.safe_load(stream)

        # Find the relevant config block:
        # - The list of dir/hist names may contain a suffix,
        #   which specifies e.g. the pt bin, jetR, or other parameters

        system = "AA" if is_AA else "pp"

        # First, check for key names in config
        if f"data_{system}{suffix}" in data:
            key = f"data_{system}{suffix}"
        elif f"data_{system}{suffix}{pt_suffix}" in data:
            key = f"data_{system}{suffix}{pt_suffix}"
        elif f"data_{system}" in data:
            key = f"data_{system}"
        else:
            # logger.info(f'data_pp_dir{suffix} not found!')
            return None

        # Get the appropriate centrality entry in the list
        if "x" in data[key]:
            x = np.array(data[key]["x"][centrality_index])
        else:
            bins = np.array(block["bins"])
            x = (bins[1:] + bins[:-1]) / 2
        n = len(x)

        if isinstance(data[key]["y"][0], list):
            if len(data[key]["y"]) > centrality_index:
                y = np.array(data[key]["y"][centrality_index])
                if "y_err" in data[key]:  # noqa: SIM108
                    y_err = np.array(data[key]["y_err"][centrality_index])
                else:
                    y_err = np.zeros(n)
            else:
                return None
        else:
            y = np.array(data[key]["y"])
            if "y_err" in data[key]:  # noqa: SIM108
                y_err = np.array(data[key]["y_err"])
            else:
                y_err = np.zeros(n)

        # Construct TGraph
        g = ROOT.TGraphAsymmErrors(n, x, y, np.zeros(n), np.zeros(n), y_err, y_err)

        return g  # noqa: RET504

    # ---------------------------------------------------------------
    # Get tgraph from the new-schema `data:` block (HEPData YAML).
    #
    # Counterpart to bins_from_data_block, but for the measured-data overlay:
    # resolves the relevant HEPData table for this observable and builds a
    # TGraphAsymmErrors from its dependent-variable values + uncertainties.
    #   - pp  -> the `spectra` block (the measured pp spectrum)
    #   - AA  -> the `ratio` block (the measured R_AA, which is what the AA
    #            plot overlays); falls back to spectra if no ratio block.
    # The y-error is the quadrature sum of every per-bin error source plus any
    # `additional_systematics` (the plot draws a single combined band, matching
    # what the old HEPData ROOT graphs encoded).
    # ---------------------------------------------------------------
    def tgraph_from_data_block(
        self, block, is_AA, sqrts, observable_type, observable, centrality, suffix="", pt_suffix="", data_block_params=None
    ):
        # Lazy import for the error-parsing helpers (reused from build_tables).
        from jetscape_analysis.data_curation import build_tables  # noqa: PLC0415

        # data_block_params selects the per-grooming / per-axis HEPData table for migrated
        # observables (same mechanism as the binning path; OBSERVABLE_EDGE_CASES A9).
        hd, match = self._resolve_data_hepdata_table(
            block, is_AA, sqrts, observable_type, observable, centrality, suffix, pt_suffix, data_block_params
        )
        if hd is None:
            return None

        # The dependent-variable index selects which measurement column to read
        # (1-based per HEPData convention; index 0 would silently pick the last block).
        try:
            index = int(match.get("index", 1))
        except (TypeError, ValueError):
            index = 0
        if index < 1:
            logger.warning(f"\tInvalid table index {match.get('index')!r} for {observable} {match.get('table')}")
            return None

        indep = (hd or {}).get("independent_variables") or []
        ind_values = indep[0].get("values") if indep else None
        try:
            dep_values = hd["dependent_variables"][index - 1]["values"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"\tdependent_variables[{index - 1}] not found for {observable}")
            return None
        axis = self._axis_from_independent_values(ind_values)
        if axis is None or not dep_values:
            logger.warning(f"\tEmpty/unusable values for {observable}")
            return None
        _edges, centers, exls, exhs = axis

        # Build the point arrays. x = bin centers (shared with the histogram
        # binning), x-errors = half bin widths, y from the dependent variable,
        # y-errors = quadrature sum of all sources.
        additional_syst = match.get("additional_systematics") or {}
        x, exl, exh, y, eyl, eyh = [], [], [], [], [], []
        for i, dv in enumerate(dep_values):
            if i >= len(centers):
                break
            val = dv.get("value")
            if val is None or (isinstance(val, str) and val.strip() in ("", "-")):
                continue
            try:
                yv = float(val)
            except (TypeError, ValueError):
                continue
            var_lo = var_hi = 0.0
            for err in dv.get("errors", []) or []:
                elo, ehi = build_tables._parse_error(err, yv)
                var_lo += elo * elo
                var_hi += ehi * ehi
            for frac in additional_syst.values():
                a = abs(yv) * float(frac)
                var_lo += a * a
                var_hi += a * a
            x.append(float(centers[i]))
            exl.append(float(exls[i]))
            exh.append(float(exhs[i]))
            y.append(yv)
            eyl.append(var_lo**0.5)
            eyh.append(var_hi**0.5)

        if not x:
            logger.warning(f"\tNo valid data points for {observable} {centrality} {suffix}")
            return None

        # Preserve the special cases from the old ROOT path (mirror bins_from_data_block):
        # drop the leading "untagged" point for zg/tg_alice.
        if observable in ["zg_alice", "tg_alice"]:
            x, exl, exh, y, eyl, eyh = (a[1:] for a in (x, exl, exh, y, eyl, eyh))
            if not x:
                return None

        n = len(x)
        return ROOT.TGraphAsymmErrors(
            n,
            np.array(x, dtype=np.float64),
            np.array(y, dtype=np.float64),
            np.array(exl, dtype=np.float64),
            np.array(exh, dtype=np.float64),
            np.array(eyl, dtype=np.float64),
            np.array(eyh, dtype=np.float64),
        )

    # ---------------------------------------------------------------
    # Truncate data tgraph to histogram binning range
    # ---------------------------------------------------------------
    def truncate_tgraph(self, g, h, is_AA=False):
        # logger.debug('truncate_tgraph')
        # logger.debug(h.GetName())
        # logger.debug(np.array(h.GetXaxis().GetXbins()))

        # Create new TGraph with number of points equal to number of histogram bins
        nBins = h.GetNbinsX()
        g_new = ROOT.TGraphAsymmErrors(nBins)
        g_new.SetName(f"{g.GetName()}_truncated")

        h_offset = 0
        for bin in range(1, nBins + 1):
            # Get histogram (x)
            h_x = h.GetBinCenter(bin)

            # Get TGraph (x,y) and errors
            gx, gy, yErrLow, yErrUp = self.get_gx_gy(g, bin - 1)

            # logger.debug(f'h_x: {h_x}')
            # logger.debug(f'gx: {gx}')
            # logger.debug(f'gy: {gy}')

            # If TGraph is offset from center of the bin, center it
            xErrLow = g.GetErrorXlow(bin - 1)
            xErrUp = g.GetErrorXhigh(bin - 1)
            if xErrLow > 0 and xErrUp > 0:
                x_min = gx - xErrLow
                x_max = gx + xErrUp
                x_center = (x_min + x_max) / 2.0
                if x_min < h_x < x_max and not np.isclose(gx, x_center):
                    gx = x_center

            # If tgraph starts below hist (e.g. when hist has min cut), try to get next tgraph point
            g_offset = 0
            while gx + 1e-8 < h_x and g_offset < g.GetN() + 1:
                g_offset += 1
                gx, gy, yErrLow, yErrUp = self.get_gx_gy(g, bin - 1 + g_offset)
            # logger.debug(f'new gx: {gx}')

            # If tgraph started above hist (see below) and we exhausted the tgraph points, skip
            if h_offset > 0 and np.isclose(gx, 0):
                continue

            # If tgraph starts above hist, try to get next hist bin
            h_offset = 0
            while gx - 1e-8 > h_x and h_offset < nBins + 1:
                h_offset += 1
                h_x = h.GetBinCenter(bin + h_offset)
                # logger.debug(f'h_x: {h_x}')
                # logger.debug(f'gx: {gx}')

            if not np.isclose(h_x, gx):
                # The histogram bin has no matching data-graph point -- e.g. the MC histogram extends
                # past the HEPData x-range (graph x reads 0 once exhausted), as for inclusive_jet/pt_cms
                # at high pt. Skip this observable's overlay/ratio gracefully and warn, rather than
                # crashing. Previously the AA arm RAISED here, aborting the entire R_AA render on the
                # first such observable (the rest of the AA R_AA plots never got produced).
                _msg = f"hist x: {h_x}, graph x: {gx}"
                logger.warning(f"truncate_tgraph: {_msg} (skipping overlay/ratio for {h.GetName()}, is_AA={is_AA})")
                return None

            g_new.SetPoint(bin - 1, gx, gy)
            g_new.SetPointError(bin - 1, 0, 0, yErrLow, yErrUp)

        return g_new

    # ---------------------------------------------------------------
    # Divide a histogram by a tgraph, point-by-point
    # ---------------------------------------------------------------
    def divide_histogram_by_tgraph(self, h, g, include_tgraph_uncertainties=True):
        # Build the ratio point-by-point over the DATA graph's points, matching each to the
        # histogram bin that contains it (h.FindBin). This is robust to a data graph that has fewer
        # points than the histogram has bins -- e.g. when the measured spectrum leaves some pT bins
        # unpublished ('-'), so the graph has gaps. (The previous index-aligned approach via
        # truncate_tgraph bailed on the whole ratio at the first such mismatch.) We emit a ratio
        # point only where the graph point falls inside a histogram bin that has nonzero content.
        g_new = ROOT.TGraphAsymmErrors()
        g_new.SetName(f"{g.GetName()}_divided")

        n_new = 0
        for i in range(g.GetN()):
            gx, gy, yErrLow, yErrUp = self.get_gx_gy(g, i)
            if gy == 0.0:
                continue  # no measured value here (e.g. an unpublished '-' bin)

            bin = h.FindBin(gx)
            if bin < 1 or bin > h.GetNbinsX():
                continue  # graph point outside the histogram range
            # Guard against mis-association: the graph x must lie within the matched bin.
            if not (h.GetXaxis().GetBinLowEdge(bin) <= gx <= h.GetXaxis().GetBinUpEdge(bin)):
                logger.warning(f"graph x: {gx} not inside its matched hist bin {bin} -- skipping point")
                continue

            h_y = h.GetBinContent(bin)
            h_error = h.GetBinError(bin)
            if h_y == 0.0:
                continue  # nothing to divide (empty MC bin)

            new_content = h_y / gy

            # Combine tgraph and histogram relative uncertainties in quadrature
            if gy > 0.0 and h_y > 0.0:
                if include_tgraph_uncertainties:
                    new_error_low = np.sqrt(pow(yErrLow / gy, 2) + pow(h_error / h_y, 2)) * new_content
                    new_error_up = np.sqrt(pow(yErrUp / gy, 2) + pow(h_error / h_y, 2)) * new_content
                else:
                    new_error_low = h_error / h_y * new_content
                    new_error_up = h_error / h_y * new_content
            else:
                new_error_low = (yErrLow / gy) * new_content
                new_error_up = (yErrUp / gy) * new_content

            # Place the ratio point at the histogram bin center (matches the MC band's x and the
            # pre-rewrite behavior) rather than the raw data x, so the lower panel stays aligned.
            g_new.SetPoint(n_new, h.GetXaxis().GetBinCenter(bin), new_content)
            g_new.SetPointError(n_new, 0, 0, new_error_low, new_error_up)
            n_new += 1

        return g_new if n_new > 0 else None

    # ---------------------------------------------------------------
    # Get points from tgraph by index
    # ---------------------------------------------------------------
    def get_gx_gy(self, g, index):
        g_x = ctypes.c_double(0)
        g_y = ctypes.c_double(0)
        g.GetPoint(index, g_x, g_y)
        yErrLow = g.GetErrorYlow(index)
        yErrUp = g.GetErrorYhigh(index)

        gx = g_x.value
        gy = g_y.value

        return gx, gy, yErrLow, yErrUp

    # ---------------------------------------------------------------
    # Divide a tgraph by a tgraph, point-by-point: g1/g2
    # NOTE: Ignore uncertainties on denominator
    # ---------------------------------------------------------------
    def divide_tgraph_by_tgraph(self, g1, g2):
        # Clone tgraph, in order to return a new one
        g_new = g1.Clone(f"{g1.GetName()}_divided")

        if g1.GetN() != g2.GetN():
            _msg = f"TGraph {g1.GetName()} has {g1.GetN()} points, but {g2.GetName()} has {g2.GetN()} points"
            raise ValueError(_msg)

        for i in range(0, g1.GetN()):  # noqa: PIE808
            # Get TGraph (x,y) and errors
            g1_x = ctypes.c_double(0)
            g1_y = ctypes.c_double(0)
            g1.GetPoint(i, g1_x, g1_y)
            y1ErrLow = g1.GetErrorYlow(i)
            y1ErrUp = g1.GetErrorYhigh(i)
            g1x = g1_x.value
            g1y = g1_y.value

            g2_x = ctypes.c_double(0)
            g2_y = ctypes.c_double(0)
            g2.GetPoint(i, g2_x, g2_y)
            g2x = g2_x.value
            g2y = g2_y.value

            if not np.isclose(g1x, g2x):
                _msg = f"TGraph {g1.GetName()} point {i} at {g1x}, but {g2.GetName()} at {g2x}"
                raise ValueError(_msg)

            new_content = g1y / g2y
            new_error_low = y1ErrLow / g1y * new_content
            new_error_up = y1ErrUp / g1y * new_content

            g_new.SetPoint(i, g1x, new_content)
            g_new.SetPointError(i, 0, 0, new_error_low, new_error_up)
        return g_new

    # -------------------------------------------------------------------------------------------
    # Set legend parameters
    # -------------------------------------------------------------------------------------------
    def setup_legend(self, leg, textSize, sep, title=""):
        leg.SetTextFont(42)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetFillColor(0)
        leg.SetMargin(0.25)
        leg.SetTextSize(textSize)
        leg.SetEntrySeparation(sep)
        leg.SetHeader(title)

    # -------------------------------------------------------------------------------------------
    def setOptions(self):
        font = 42

        ROOT.gStyle.SetFrameBorderMode(0)
        ROOT.gStyle.SetFrameFillColor(0)
        ROOT.gStyle.SetCanvasBorderMode(0)
        ROOT.gStyle.SetPadBorderMode(0)
        ROOT.gStyle.SetPadColor(10)
        ROOT.gStyle.SetCanvasColor(10)
        ROOT.gStyle.SetTitleFillColor(10)
        ROOT.gStyle.SetTitleBorderSize(1)
        ROOT.gStyle.SetStatColor(10)
        ROOT.gStyle.SetStatBorderSize(1)
        ROOT.gStyle.SetLegendBorderSize(1)

        ROOT.gStyle.SetDrawBorder(0)
        ROOT.gStyle.SetTextFont(font)
        ROOT.gStyle.SetStatFont(font)
        ROOT.gStyle.SetStatFontSize(0.05)
        ROOT.gStyle.SetStatX(0.97)
        ROOT.gStyle.SetStatY(0.98)
        ROOT.gStyle.SetStatH(0.03)
        ROOT.gStyle.SetStatW(0.3)
        ROOT.gStyle.SetTickLength(0.02, "y")
        ROOT.gStyle.SetEndErrorSize(3)
        ROOT.gStyle.SetLabelSize(0.05, "xyz")
        ROOT.gStyle.SetLabelFont(font, "xyz")
        ROOT.gStyle.SetLabelOffset(0.01, "xyz")
        ROOT.gStyle.SetTitleFont(font, "xyz")
        ROOT.gStyle.SetTitleOffset(1.2, "xyz")
        ROOT.gStyle.SetTitleSize(0.045, "xyz")
        ROOT.gStyle.SetMarkerSize(1)
        ROOT.gStyle.SetPalette(1)

        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)
