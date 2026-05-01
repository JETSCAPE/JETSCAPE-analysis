from __future__ import annotations

import copy
import io
import logging
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yaml

from jetscape_analysis.data_curation import data as data_module
from jetscape_analysis.data_curation import observable as obs_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# One row in the combinations editor
RowData = dict[str, Any]  # keys: param_key, parameters, table, index, systematics_names, additional_systematics

# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

if "observables" not in st.session_state:
    st.session_state.observables = {}
if "combo_data" not in st.session_state:
    st.session_state.combo_data = {}
if "yaml_output" not in st.session_state:
    st.session_state.yaml_output = ""


# ---------------------------------------------------------------------------
# Observable loading
# ---------------------------------------------------------------------------


def _parse_observables_from_dict(config: dict[str, Any], sqrt_s: int) -> dict[str, obs_module.Observable]:
    """Parse observables from an already-loaded config dict."""
    observable_classes: list[str] = []
    found = False
    for k in config:
        if "hadron" in k or found:
            observable_classes.append(k)
            found = True

    result: dict[str, obs_module.Observable] = {}
    for observable_class in observable_classes:
        for observable_key, observable_info in config[observable_class].items():
            key = f"{sqrt_s}_{observable_class}_{observable_key}"
            result[key] = obs_module.Observable(
                sqrt_s=sqrt_s,
                observable_class=observable_class,
                name=observable_key,
                config=observable_info,
            )
    return result


def setup_observables(uploaded_file: Any = None) -> None:
    """Load observables. Re-runs if an uploaded file is provided."""
    if uploaded_file is not None:
        content = uploaded_file.read()
        config = yaml.safe_load(io.BytesIO(content))
        sqrt_s = int(config.get("sqrt_s", 0))
        st.session_state.observables = _parse_observables_from_dict(config, sqrt_s)
    elif not st.session_state.observables:
        _here = Path(__file__).parent
        st.session_state.observables = obs_module.read_observables_from_all_config(
            jetscape_analysis_config_path=_here.parent.parent / "config"
        )


# ---------------------------------------------------------------------------
# Parameter combination helpers
# ---------------------------------------------------------------------------


def _encode_params(params: dict[str, Any]) -> dict[str, str]:
    """Convert a dict of encode_name -> ParameterSpec values to encode_name -> encoded string."""
    return {k: (v.encode() if hasattr(v, "encode") else str(v)) for k, v in params.items()}


def _param_key(encoded: dict[str, str]) -> str:
    """Stable string key from an encoded parameter dict (for matching)."""
    return "|".join(f"{k}={v}" for k, v in sorted(encoded.items()))


def get_storage_key(obs_key: str, collision_system: str, histogram_name: str) -> str:
    return f"{obs_key}|{collision_system}|{histogram_name}"


# ---------------------------------------------------------------------------
# Loading / initializing combination data
# ---------------------------------------------------------------------------


def _normalize_param_value(v: Any) -> str:
    """Normalize a YAML parameter value (str, list, int, float) to a string for matching."""
    if isinstance(v, list) and len(v) == 1:
        return str(v[0])
    return str(v) if not isinstance(v, str) else v


def _load_existing_entries(
    obs: obs_module.Observable,
    collision_system: str,
    histogram_name: str,
) -> list[RowData]:
    """Parse the existing `data` section into a flat list of entries with normalized parameters."""
    data_section = obs.config.get("data", {})
    raw_tables = data_section.get(collision_system, {}).get("hepdata", {}).get(histogram_name, {}).get("tables", [])
    if not raw_tables:
        return []

    result: list[RowData] = []
    expanded = data_module.expand_parameter_combinations_into_individual_configs(raw_tables)
    for entry in expanded:
        norm_params = {k: _normalize_param_value(v) for k, v in entry.get("parameters", {}).items()}
        result.append(
            {
                "parameters": norm_params,
                "table": entry.get("table", ""),
                "index": str(entry.get("index", "")),
                "systematics_names": dict(entry.get("systematics_names", {})),
                "additional_systematics": dict(entry.get("additional_systematics", {})),
            }
        )
    return result


def _find_matching_entry(
    combo_encoded: dict[str, str],
    existing_entries: list[RowData],
) -> RowData | None:
    """Find the first existing entry whose parameters are a subset of the combo's parameters."""
    for entry in existing_entries:
        existing_params = entry["parameters"]
        if all(combo_encoded.get(k) == v for k, v in existing_params.items()):
            return entry
    return None


def init_combo_data(obs: obs_module.Observable, collision_system: str, histogram_name: str) -> list[RowData]:
    """Build the initial list of RowData from observable parameters + existing config."""
    existing_entries = _load_existing_entries(obs, collision_system, histogram_name)

    all_params = obs.parameters()
    param_names = [p.encode_name for p in all_params]
    values_lists = [p.values for p in all_params]

    rows: list[RowData] = []
    for combo_values in product(*values_lists):
        raw_params = dict(zip(param_names, combo_values, strict=True))
        encoded = _encode_params(raw_params)

        existing = _find_matching_entry(encoded, existing_entries)
        rows.append(
            {
                "param_key": _param_key(encoded),
                "parameters": encoded,
                "table": existing["table"] if existing else "",
                "index": existing["index"] if existing else "",
                "systematics_names": dict(existing["systematics_names"]) if existing else {},
                "additional_systematics": dict(existing["additional_systematics"]) if existing else {},
            }
        )

    return rows


def get_combo_data(
    obs_key: str, obs: obs_module.Observable, collision_system: str, histogram_name: str
) -> list[RowData]:
    key = get_storage_key(obs_key, collision_system, histogram_name)
    if key not in st.session_state.combo_data:
        st.session_state.combo_data[key] = init_combo_data(obs, collision_system, histogram_name)
    return st.session_state.combo_data[key]


def save_combo_data(obs_key: str, collision_system: str, histogram_name: str, data: list[RowData]) -> None:
    key = get_storage_key(obs_key, collision_system, histogram_name)
    st.session_state.combo_data[key] = data


# ---------------------------------------------------------------------------
# HEPData systematics fetching
# ---------------------------------------------------------------------------


def get_hepdata_uncertainties(
    url: str,
    table_name: str,
    entry_index: int = 0,
    version: int = 1,
) -> dict[str, str]:
    """Download a HEPData table and return a dict of uncertainty_name -> uncertainty_name."""
    import urllib.parse  # noqa: PLC0415

    if "ins" not in url:
        msg = "Invalid HEPData URL format"
        raise ValueError(msg)
    record_id = url.rsplit("ins", maxsplit=1)[-1].split("/", maxsplit=1)[0]
    api_url = f"https://www.hepdata.net/download/table/ins{record_id}/{urllib.parse.quote(table_name)}/{version}/json"

    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("name") != table_name:
        msg = f"Wrong table name. Got: {data.get('name')!r}"
        raise ValueError(msg)

    table_values = data.get("values")
    if table_values is None:
        msg = f"Table '{table_name}' not found"
        raise KeyError(msg)

    uncertainties: list[str] = []
    values = table_values
    if entry_index >= len(values):
        msg = f"Entry index {entry_index} out of range. Table has {len(values)} entries."
        raise IndexError(msg)
    entry = values[entry_index]
    for error_type in ("errors", "uncertainties"):
        for error in entry.get(error_type, []):
            if "label" in error:
                uncertainties.append(error["label"])

    seen: set[str] = set()
    unique: list[str] = []
    for u in uncertainties:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return {u: u for u in unique}


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def build_tables_yaml(rows: list[RowData]) -> list[dict[str, Any]]:
    """Convert list of RowData to the `tables:` list format."""
    tables = []
    for row in rows:
        entry: dict[str, Any] = {
            "parameters": dict(row["parameters"]),
            "systematics_names": dict(row["systematics_names"]),
            "additional_systematics": dict(row["additional_systematics"]),
            "table": row["table"],
            "index": row["index"],
        }
        tables.append(entry)
    return tables


def generate_data_yaml(
    obs: obs_module.Observable,
    all_combo_data: dict[str, list[RowData]],
    obs_key: str,
) -> str:
    """Generate the full `data:` section YAML for an observable."""
    # Start with a deep copy of the existing data section (preserves record, axes, etc.)
    existing_data = copy.deepcopy(obs.config.get("data", {}))

    # Overwrite the tables for each (collision_system, histogram_name) combination
    for storage_key, rows in all_combo_data.items():
        parts = storage_key.split("|")
        if parts[0] != obs_key:
            continue
        _, collision_system, histogram_name = parts

        tables_yaml = build_tables_yaml(rows)
        # Navigate/create the nested structure
        existing_data.setdefault(collision_system, {})
        existing_data[collision_system].setdefault("hepdata", {})
        existing_data[collision_system]["hepdata"].setdefault(histogram_name, {})
        existing_data[collision_system]["hepdata"][histogram_name]["tables"] = tables_yaml

    return yaml.dump({"data": existing_data}, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Systematics dict editor
# ---------------------------------------------------------------------------


def edit_dict_interface(
    dict_data: dict[str, Any],
    key_prefix: str,
    value_type: str = "str",
) -> None:
    """Inline editor for a dict (str->str or str->float)."""
    to_remove: list[str] = []
    for i, (k, v) in enumerate(list(dict_data.items())):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            new_k = st.text_input("Key", value=k, key=f"{key_prefix}_k_{i}", label_visibility="collapsed")
        with c2:
            if value_type == "float":
                new_v = st.number_input(
                    "Val",
                    value=float(v) if v else 0.0,
                    key=f"{key_prefix}_v_{i}",
                    label_visibility="collapsed",
                    format="%.6f",
                )
            else:
                new_v = st.text_input("Val", value=str(v), key=f"{key_prefix}_v_{i}", label_visibility="collapsed")
        with c3:
            if st.button("✕", key=f"{key_prefix}_del_{i}"):
                to_remove.append(k)
        if new_k != k:
            dict_data.pop(k, None)
            if new_k:
                dict_data[new_k] = new_v
        elif new_k:
            dict_data[new_k] = new_v
    for k in to_remove:
        dict_data.pop(k, None)

    # Add new entry
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        add_k = st.text_input("New key", key=f"{key_prefix}_add_k", placeholder="key", label_visibility="collapsed")
    with c2:
        if value_type == "float":
            add_v: Any = st.number_input(
                "New val", value=0.0, key=f"{key_prefix}_add_v", label_visibility="collapsed", format="%.6f"
            )
        else:
            add_v = st.text_input(
                "New val", key=f"{key_prefix}_add_v", placeholder="value", label_visibility="collapsed"
            )
    with c3:
        # NOTE: Not sure if these can be safely combined on the same line and running short on time,
        #       so just leaving them separate for now.
        if st.button("＋", key=f"{key_prefix}_add_btn"):  # noqa: RUF001, SIM102
            if add_k:
                dict_data[add_k] = add_v
                st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    st.title("Configure HEPData Sources")

    # --- Sidebar: file upload ---
    with st.sidebar:
        st.header("Observable source")
        uploaded = st.file_uploader("Upload a STAT YAML file (or use repo defaults)", type=["yaml", "yml"])
        if uploaded:
            if st.button("Load uploaded file"):
                setup_observables(uploaded_file=uploaded)
                st.session_state.combo_data = {}  # reset assignments
                st.rerun()
        else:
            setup_observables()

        st.divider()
        st.caption(f"Loaded {len(st.session_state.observables)} observables")

    observables = st.session_state.observables
    if not observables:
        st.error("No observables loaded.")
        return

    # --- Observable selector ---
    obs_keys = list(observables.keys())
    obs_key = st.selectbox(
        "Observable (sqrt_s · experiment · class · name)",
        obs_keys,
        format_func=lambda k: (
            f"{observables[k].sqrt_s} · {observables[k].experiment} · "
            f"{observables[k].observable_class} · {observables[k].display_name} "
            f"({observables[k].internal_name_without_experiment})"
        ),
    )
    obs = observables[obs_key]

    # --- Collision system + histogram type ---
    col1, col2 = st.columns(2)
    with col1:
        collision_system = st.selectbox("Collision system", ["AA", "pp"])
    with col2:
        histogram_options = ["spectra", "ratio"] if collision_system == "AA" else ["spectra"]
        histogram_name = st.selectbox("Histogram type", histogram_options, accept_new_options=True)

    # --- Load current combo data ---
    current_data = get_combo_data(obs_key, obs, collision_system, histogram_name)

    if not current_data:
        st.warning("No parameter combinations found for this observable.")
        return

    st.divider()

    # --- Parameters overview dataframe ---
    st.subheader("Parameter combinations")
    param_names = list(current_data[0]["parameters"].keys())
    df_display = pd.DataFrame(
        [{**row["parameters"], "Table": row["table"], "Index": row["index"]} for row in current_data]
    )
    # Show a static overview (not editable) for reference
    st.dataframe(df_display, width="stretch", hide_index=False)

    st.divider()

    # --- Bulk editor: table/index via data_editor ---
    st.subheader("Edit table and index assignments")
    st.caption("Parameter columns are read-only. Edit Table and Index directly.")

    df_edit = pd.DataFrame(
        [{**row["parameters"], "Table": row["table"], "Index": row["index"]} for row in current_data]
    )

    col_config = {p: st.column_config.TextColumn(p, disabled=True) for p in param_names}
    col_config["Table"] = st.column_config.TextColumn("HEPData Table", help='e.g. "Table 4"')
    col_config["Index"] = st.column_config.TextColumn("Index", help="Row index within the table (1-indexed)")

    edited_df = st.data_editor(
        df_edit,
        column_config=col_config,
        width="stretch",
        hide_index=False,
        key=f"editor_{obs_key}_{collision_system}_{histogram_name}",
    )

    # Sync edits back to current_data
    for i, row in edited_df.iterrows():
        current_data[i]["table"] = row["Table"] or ""
        current_data[i]["index"] = row["Index"] or ""

    save_combo_data(obs_key, collision_system, histogram_name, current_data)

    st.divider()

    # --- Systematics editor (per combination, in expanders) ---
    st.subheader("Systematics")
    hepdata_url = obs.config.get("urls", {}).get("hepdata", "")

    for i, row in enumerate(current_data):
        param_label = ", ".join(f"{k}={v}" for k, v in row["parameters"].items())
        with st.expander(f"Combination {i}: {param_label}", expanded=False):
            # Load systematics from HEPData
            if row["table"]:
                c1, c2 = st.columns([1, 3])
                with c1:
                    if st.button("Load from HEPData", key=f"load_sys_{i}"):
                        if hepdata_url:
                            try:
                                idx = int(row["index"]) - 1 if row["index"] else 0
                                loaded = get_hepdata_uncertainties(
                                    url=hepdata_url,
                                    table_name=row["table"],
                                    entry_index=max(0, idx),
                                )
                                row["systematics_names"].update(loaded)
                                save_combo_data(obs_key, collision_system, histogram_name, current_data)
                                st.success(f"Loaded {len(loaded)} uncertainties")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        else:
                            st.warning("No HEPData URL found for this observable.")
                with c2:
                    st.caption("Fetches uncertainty labels from the HEPData table for the given row index.")
            else:
                st.info("Set a Table name to enable HEPData uncertainty loading.")

            st.markdown("**Systematics names** (HEPData label → analysis label)")
            col_headers = st.columns([2, 2, 1])
            col_headers[0].caption("HEPData label")
            col_headers[1].caption("Analysis label")
            edit_dict_interface(row["systematics_names"], f"sys_{i}", "str")

            st.markdown("**Additional systematics** (name → value)")
            col_headers2 = st.columns([2, 2, 1])
            col_headers2[0].caption("Name")
            col_headers2[1].caption("Value")
            edit_dict_interface(row["additional_systematics"], f"addsys_{i}", "float")

    save_combo_data(obs_key, collision_system, histogram_name, current_data)

    st.divider()

    # --- Export ---
    st.subheader("Export")
    st.caption(
        "Generates the `data:` section for this observable. "
        "Covers all collision systems and histogram types you have edited in this session."
    )

    if st.button("Generate YAML", type="primary"):
        yaml_str = generate_data_yaml(obs, st.session_state.combo_data, obs_key)
        st.session_state.yaml_output = yaml_str

    if st.session_state.yaml_output:
        st.code(st.session_state.yaml_output, language="yaml")
        st.download_button(
            label="Download YAML",
            data=st.session_state.yaml_output,
            file_name=f"data_{obs_key}.yaml",
            mime="text/yaml",
        )


if __name__ == "__main__":
    main()
