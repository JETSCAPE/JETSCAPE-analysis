"""Build curated data tables (.dat format) from the STAT_{sqrt_s}.yaml config and HEPData files.

Output follows the JetScape data-file format specification v1.0 (Note 1423):
    # Version 1.0
    # DOI <url>
    # Source <hepdata url> <table refs>
    # System PbPb2760
    # Centrality 0 10
    # XY PT RAA
    # Label xmin xmax y stat,low stat,high sys,<name>,low sys,<name>,high ...
    <xmin> <xmax> <y> <stat_lo> <stat_hi> ...

Only AA/ratio (RAA) blocks are written.  One output file per (centrality, jet_R) combination.
Filename convention: Data_{exp}_{system}_{measurement}_{cent}_{year}.dat

.. codeauthor:: auto-generated
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from jetscape_analysis.data_curation import data as data_mod
from jetscape_analysis.data_curation import hepdata_utils, observable

logger = logging.getLogger(__name__)

# Mapping from "{observable_class}/{internal_name_without_experiment}" to measurement tag prefix.
# For jet observables, the jet_R suffix is appended dynamically (e.g. RAAJetR04).
_MEASUREMENT_TAG: dict[str, str] = {
    "hadron/pt_ch": "RAACharged",
    "hadron/pt_pi": "RAAPi",
    "hadron/pt_pi0": "RAAPi0",
    "inclusive_jet/pt": "RAAJet",
}


def _system_tag(sqrt_s: int) -> str:
    if sqrt_s == 200:
        return f"AuAu{sqrt_s}"
    return f"PbPb{sqrt_s}"


def _centrality_tag(cent: observable.CentralitySpec) -> str:
    return f"{_fmt_int(cent.low)}to{_fmt_int(cent.high)}"


def _fmt_int(x: float) -> str:
    return f"{int(x)}" if float(x).is_integer() else f"{x}"


def _measurement_tag(obs: observable.Observable, combo: dict[str, Any]) -> str:
    key = f"{obs.observable_class}/{obs.internal_name_without_experiment}"
    base = _MEASUREMENT_TAG.get(key, obs.internal_name_without_experiment.upper())
    jet_r = combo.get("jet_R")
    if isinstance(jet_r, observable.JetRSpec):
        base = f"{base}R{int(round(jet_r.R * 10)):02d}"
    return base


_INSPIRE_YEAR_CACHE: dict[int, str] = {}


def _year_from_inspire(inspire_hep_id: int) -> str:
    """Fetch the paper publication year from the Inspire HEP API. Cached per record ID."""
    if inspire_hep_id in _INSPIRE_YEAR_CACHE:
        return _INSPIRE_YEAR_CACHE[inspire_hep_id]
    try:
        import requests  # noqa: PLC0415

        resp = requests.get(
            f"https://inspirehep.net/api/literature/{inspire_hep_id}",
            params={"fields": "publication_info,earliest_date"},
            timeout=10,
        )
        resp.raise_for_status()
        meta = resp.json().get("metadata", {})
        # Prefer publication year; fall back to earliest_date
        for pub in meta.get("publication_info", []):
            if "year" in pub:
                year = str(pub["year"])
                _INSPIRE_YEAR_CACHE[inspire_hep_id] = year
                return year
        earliest = meta.get("earliest_date", "")
        if earliest:
            year = earliest.split("-")[0]
            _INSPIRE_YEAR_CACHE[inspire_hep_id] = year
            return year
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Inspire API lookup failed for ins{inspire_hep_id}: {exc}")
    _INSPIRE_YEAR_CACHE[inspire_hep_id] = "unknown"
    return "unknown"


def _year_from_hepdata(hd_info: hepdata_utils.HEPDataInfo, base_data_dir: Path) -> str:
    """Extract year, preferring HEPData submission.yaml, falling back to Inspire API.

    Some HEPData records put ``dateupdated`` in the first meta doc, others in a later one,
    and a few omit it altogether. We try all docs first; if nothing, we ask Inspire.
    """
    sub = Path(base_data_dir) / "data" / hd_info.directory / "submission.yaml"
    try:
        with sub.open() as f:
            metas = list(yaml.safe_load_all(f))
        for m in metas:
            if isinstance(m, dict) and "dateupdated" in m:
                date = m["dateupdated"]
                # Format "24/04/2015 07:16:36" → "2015"
                return date.split()[0].split("/")[-1]
    except Exception:  # noqa: BLE001
        pass
    return _year_from_inspire(hd_info.identifier.inspire_hep_id)


def _to_abs_error(val: Any, y: float) -> float:
    """Convert a HEPData error value (float, "3.1%", etc.) to an absolute magnitude."""
    if val is None:
        return 0.0
    if isinstance(val, str) and val.endswith("%"):
        return abs(y) * float(val.rstrip("%")) / 100.0
    return float(val)


def _parse_error(err_entry: dict[str, Any], y: float) -> tuple[float, float]:
    """Return (low, high) magnitudes. Symmetric → duplicated; asymmetric → (|minus|, |plus|)."""
    if "symerror" in err_entry:
        v = abs(_to_abs_error(err_entry["symerror"], y))
        return v, v
    if "asymerror" in err_entry:
        ae = err_entry["asymerror"]
        lo = abs(_to_abs_error(ae.get("minus"), y))
        hi = abs(_to_abs_error(ae.get("plus"), y))
        return lo, hi
    return 0.0, 0.0


def _decode_centrality(raw: str) -> observable.CentralitySpec:
    lo, hi = raw.split("_")
    return observable.CentralitySpec(low=float(lo), high=float(hi))


def _decode_pt_range(raw: str) -> observable.PtSpec | None:
    try:
        lo, hi = raw.split("_")
        return observable.PtSpec(low=float(lo), high=float(hi) if float(hi) >= 0 else None)
    except (ValueError, AttributeError):
        return None


def _decode_jet_r(raw: str) -> observable.JetRSpec | None:
    try:
        return observable.JetRSpec(R=float(raw))
    except (ValueError, TypeError):
        return None


def _find_centrality(combo: dict[str, Any]) -> observable.CentralitySpec | None:
    for k, v in combo.items():
        if "centrality" in str(k).lower():
            return _decode_centrality(v) if isinstance(v, str) else v
    return None


def _find_jet_r(combo: dict[str, Any]) -> observable.JetRSpec | None:
    for k, v in combo.items():
        if "jet_r" in str(k).lower() or str(k) == "R":
            return _decode_jet_r(v) if isinstance(v, str) else v
    return None


def _find_pt_range(combo: dict[str, Any]) -> observable.PtSpec | None:
    """Return the PtSpec for filtering HEPData bins (if any in the combination)."""
    for k, v in combo.items():
        if str(k) == "pt":
            return _decode_pt_range(v) if isinstance(v, str) else v
    return None


def _systematic_column_prefix(canonical: str) -> str:
    """Return the column-name prefix for a systematic source.

    If the canonical name is already ``sys`` (i.e. a lumped HEPData systematic kept as-is),
    we avoid the degenerate ``sys,sys,low`` double prefix and write just ``sys,low``.
    """
    if canonical == "sys":
        return "sys"
    return f"sys,{canonical}"


def _expand_ratio_entries(obs: observable.Observable) -> list[dict[str, Any]]:
    """Return expanded ratio-block configs (raw dicts), or [] if missing/empty.

    We do *not* convert to HEPDataEntry here because the current registry has an
    unresolved key mismatch for ``jet_R`` (registered as ``R``). Returning raw
    dicts lets us decode parameter values by hand in the writer.
    """
    ratio_cfg = (
        obs.config.get("data", {})
        .get("AA", {})
        .get("hepdata", {})
        .get("ratio")
    )
    if not ratio_cfg:
        return []
    tables = ratio_cfg.get("tables", [])
    expanded = data_mod.expand_parameter_combinations_into_individual_configs(tables)
    entries: list[dict[str, Any]] = []
    for e in expanded:
        if not e.get("table") or e.get("index") in (None, ""):
            logger.debug(f"Skipping empty ratio entry for {obs.observable_str}: {e}")
            continue
        # Skip unfilled blocks (systematics_names still empty) — nothing to write.
        if not e.get("systematics_names") and not e.get("additional_systematics"):
            logger.debug(
                f"Skipping unfilled ratio entry for {obs.observable_str} (empty systematics_names)"
            )
            continue
        entries.append(e)
    return entries


def _format_number(x: float) -> str:
    return f"{x:.6g}"


def write_data_table(
    obs: observable.Observable,
    entry: dict[str, Any],
    hd_info: hepdata_utils.HEPDataInfo,
    base_data_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Write a single Data_*.dat file for one expanded ratio-block entry.

    Returns the output path, or None if the entry could not be written.
    """
    table_name: str = entry["table"]
    table_index: str = str(entry["index"])
    systematics_names: dict[str, str] = entry.get("systematics_names") or {}
    additional_syst: dict[str, float] = entry.get("additional_systematics") or {}
    params: dict[str, Any] = entry.get("parameters") or {}

    # Locate the HEPData YAML file
    table_rel_filename = hd_info.tables_to_filenames.get(table_name)
    if table_rel_filename is None:
        logger.warning(
            f"{obs.observable_str}: '{table_name}' not in hepdata record {hd_info.identifier}; skipping"
        )
        return None
    yaml_path = Path(base_data_dir) / "data" / hd_info.directory / table_rel_filename
    with yaml_path.open() as f:
        hd = yaml.safe_load(f)

    # Independent variable → bin edges (prefer xmin/xmax, fall back to x)
    ind = hd["independent_variables"][0]
    has_bin_edges = all(
        ("low" in b and "high" in b and b.get("low") is not None and b.get("high") is not None)
        for b in ind["values"]
    )

    # Dependent variable (select by index, 1-based per HEPData convention)
    try:
        dv = hd["dependent_variables"][int(table_index) - 1]
    except (IndexError, ValueError):
        logger.warning(
            f"{obs.observable_str}: table_index={table_index} out of range for {table_name}; skipping"
        )
        return None

    # Build systematic columns in config order: skip "stat" (written separately), use canonical names.
    per_bin_syst_cols: list[tuple[str, str]] = []  # (hepdata_label, canonical_name)
    stat_label: str | None = None
    for hep_label, canonical in systematics_names.items():
        if canonical == "stat":
            stat_label = hep_label
            continue
        per_bin_syst_cols.append((hep_label, canonical))
    additional_cols: list[tuple[str, float]] = list(additional_syst.items())

    # Determine filename components
    cent = _find_centrality(params)
    if cent is None:
        logger.warning(f"{obs.observable_str}: no centrality in entry params {params}; skipping")
        return None
    exp = obs.experiment
    system = _system_tag(obs.sqrt_s)
    # _measurement_tag needs JetRSpec (if any) — build a decoded combo
    decoded_combo: dict[str, Any] = {}
    for k, v in params.items():
        if "jet_r" in str(k).lower() or str(k) == "R":
            jr = _decode_jet_r(v) if isinstance(v, str) else v
            if jr is not None:
                decoded_combo[k] = jr
    measurement = _measurement_tag(obs, decoded_combo)
    cent_tag = _centrality_tag(cent)
    year = _year_from_hepdata(hd_info, base_data_dir)

    fn = f"Data_{exp}_{system}_{measurement}_{cent_tag}_{year}.dat"
    out_dir = output_dir / str(obs.sqrt_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fn

    hepdata_url = f"https://www.hepdata.net/record/ins{hd_info.identifier.inspire_hep_id}"
    inspire_url = f"https://inspirehep.net/literature/{hd_info.identifier.inspire_hep_id}"

    # Y-label: for the ratio block it's always RAA
    y_label = "RAA"
    # X-label: take from observable's internal name (pt → PT, etc.)
    x_token = obs.internal_name_without_experiment.split("_")[0].upper()

    # Write file
    with out_path.open("w") as f:
        f.write("# Version 1.0\n")
        f.write(f"# DOI {hepdata_url}\n")
        f.write(f"# InspireHEP {inspire_url}\n")
        f.write(f"# Source {hepdata_url} {table_name}\n")
        f.write(f"# System {system}\n")
        f.write(f"# Centrality {_fmt_int(cent.low)} {_fmt_int(cent.high)}\n")
        f.write(f"# XY {x_token} {y_label}\n")

        # Label header
        header_cols = (["xmin", "xmax", "y"] if has_bin_edges else ["x", "y"]) + [
            "stat,low",
            "stat,high",
        ]
        for _, canonical in per_bin_syst_cols:
            prefix = _systematic_column_prefix(canonical)
            header_cols += [f"{prefix},low", f"{prefix},high"]
        for canonical, _ in additional_cols:
            prefix = _systematic_column_prefix(canonical)
            header_cols += [f"{prefix},low", f"{prefix},high"]
        f.write("# Label " + " ".join(header_cols) + "\n")

        # pT range filter (if the config entry specifies one)
        pt_range = _find_pt_range(params)

        # Rows
        n_written = 0
        for ind_entry, v in zip(ind["values"], dv["values"], strict=False):
            y_raw = v.get("value")
            # Skip HEPData "missing" sentinels like "-", "", None
            if y_raw is None or (isinstance(y_raw, str) and not y_raw.replace(".", "").replace("-", "").replace("e", "").replace("+", "").strip()):
                continue
            try:
                y = float(y_raw)
            except (TypeError, ValueError):
                logger.debug(f"{obs.observable_str} {entry.table}: skipping non-numeric value {y_raw!r}")
                continue
            # pT range filter
            if pt_range is not None and has_bin_edges:
                bin_lo, bin_hi = float(ind_entry["low"]), float(ind_entry["high"])
                if bin_hi <= pt_range.low or (pt_range.high is not None and bin_lo >= pt_range.high):
                    continue

            # Locate errors by HEPData label
            err_by_label = {e.get("label", ""): e for e in v.get("errors", [])}

            row: list[str] = []
            if has_bin_edges:
                row += [_format_number(ind_entry["low"]), _format_number(ind_entry["high"])]
            else:
                row += [_format_number(ind_entry.get("value", 0))]
            row += [_format_number(y)]

            # Stat
            if stat_label and stat_label in err_by_label:
                lo, hi = _parse_error(err_by_label[stat_label], y)
            else:
                lo, hi = 0.0, 0.0
            row += [_format_number(lo), _format_number(hi)]

            # Per-bin systematics in config order
            for hep_label, _ in per_bin_syst_cols:
                if hep_label in err_by_label:
                    lo, hi = _parse_error(err_by_label[hep_label], y)
                else:
                    lo, hi = 0.0, 0.0
                row += [_format_number(lo), _format_number(hi)]

            # Additional systematics (global, constant across bins) — frac × |y|
            for _, frac in additional_cols:
                abs_err = abs(y) * float(frac)
                row += [_format_number(abs_err), _format_number(abs_err)]

            f.write(" ".join(row) + "\n")
            n_written += 1

    logger.info(f"Wrote {n_written} rows to {out_path}")
    return out_path


def build_tables_for_config(config_path: Path, output_dir: Path, base_data_dir: Path | None = None) -> list[Path]:
    """Build all RAA data tables for observables in the given STAT_{sqrt_s}.yaml config."""
    if base_data_dir is None:
        base_data_dir = hepdata_utils.BASE_DATA_DIR
    base_data_dir = Path(base_data_dir)

    # Derive sqrt_s from filename (e.g. STAT_2760.yaml → 2760)
    stem = Path(config_path).stem
    sqrt_s_str = stem.split("_")[-1]
    try:
        sqrt_s = int(sqrt_s_str)
    except ValueError as exc:
        msg = f"Cannot infer sqrt_s from config filename '{config_path.name}'"
        raise ValueError(msg) from exc

    with Path(config_path).open() as f:
        cfg = yaml.safe_load(f)

    # Identify observable classes (same logic as observable.read_observables_from_config)
    observable_classes: list[str] = []
    started = False
    for k in cfg:
        if "hadron" in k or started:
            observable_classes.append(k)
            started = True

    output_paths: list[Path] = []
    for obs_class in observable_classes:
        for obs_name, obs_cfg in cfg[obs_class].items():
            if not obs_cfg.get("enabled", False):
                continue
            obs = observable.Observable(
                sqrt_s=sqrt_s,
                observable_class=obs_class,
                name=obs_name,
                config=obs_cfg,
            )
            entries = _expand_ratio_entries(obs)
            if not entries:
                logger.debug(f"{obs.observable_str}: no filled ratio entries; skipping")
                continue

            # Retrieve HEPData info once per (observable, inspire_hep_id).
            # Since all ratio entries for one observable share the same record, we fetch once.
            inspire_id = obs_cfg["data"]["AA"]["hepdata"]["record"]["inspire_hep_id"]
            version = obs_cfg["data"]["AA"]["hepdata"]["record"].get("version", -1)
            hd_info = hepdata_utils.retrieve_observable_hepdata(
                observable_str_as_path=obs.observable_str_as_path,
                inspire_hep_id=inspire_id,
                version=version,
                base_dir=base_data_dir,
            )

            for entry in entries:
                out = write_data_table(
                    obs=obs,
                    entry=entry,
                    hd_info=hd_info,
                    base_data_dir=base_data_dir,
                    output_dir=output_dir,
                )
                if out is not None:
                    output_paths.append(out)

    logger.info(f"Wrote {len(output_paths)} data tables to {output_dir}/{sqrt_s}")
    return output_paths


def main() -> None:
    from jetscape_analysis.base import helpers  # noqa: PLC0415

    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="Path to STAT_{sqrt_s}.yaml")
    parser.add_argument("--output", type=Path, default=Path("tables"), help="Output directory (default: ./tables)")
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=None,
        help="Base dir for HEPData archives (default: hepdata_utils.BASE_DATA_DIR)",
    )
    args = parser.parse_args()

    build_tables_for_config(
        config_path=args.config,
        output_dir=args.output,
        base_data_dir=args.base_data_dir,
    )


if __name__ == "__main__":
    main()
