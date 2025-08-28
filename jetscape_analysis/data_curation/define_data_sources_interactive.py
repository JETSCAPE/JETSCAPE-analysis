from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from jetscape_analysis.data_curation import observable

logger = logging.getLogger(__name__)

# Initialize session state
# Containing the existing observables
if "observables" not in st.session_state:
    st.session_state.observables = {}
if "combinations_data" not in st.session_state:
    st.session_state.combinations_data = {}

# Configuration data
PARAM_CONFIG: dict[str, list[Any]] = {
    "jet_R": [0.2, 0.3],
    "soft_drop": ["z_cut_02_beta_0"],
    "centrality": [[0, 10], [10, 20]],
    "jet_pt": [[100, 200], [200, 300]],
}

PARAM_ORDER: list[str] = ["jet_R", "soft_drop", "centrality", "jet_pt"]

EXISTING_LOOKUP: dict[str, dict[str, dict[str, dict[str, int]]]] = {
    "AA": {
        "spectra": {
            '{"centrality":[0,10],"jet_R":0.2,"jet_pt":[100,200],"soft_drop":"z_cut_02_beta_0"}': {
                "entry": 0,
                "table": 15,
            }
        }
    },
    "pp": {
        "spectra": {
            '{"centrality":[10,20],"jet_R":0.3,"jet_pt":[200,300],"soft_drop":"z_cut_02_beta_0"}': {
                "entry": 1,
                "table": 12,
            }
        }
    },
}

CombinationData = dict[str, Any]
ParameterDict = dict[str, Any]
SystematicsDict = dict[str, str]
AdditionalSystematicsDict = dict[str, float]


def load_hepdata_file(filename: Path, table: str | int, index: int) -> SystematicsDict:
    """Placeholder function - you'll implement this"""
    # This should return a dictionary of systematic uncertainties
    # For now, returning dummy data
    return {"stat": "0.05", "syst_jet_energy": "0.03", "syst_tracking": "0.02"}


def generate_combinations() -> list[ParameterDict]:
    """Generate all parameter combinations"""
    param_values = [PARAM_CONFIG[param] for param in PARAM_ORDER]
    combinations: list[ParameterDict] = []

    for combo in product(*param_values):
        param_dict = dict(zip(PARAM_ORDER, combo, strict=True))
        combinations.append(param_dict)

    return combinations


def get_storage_key(collision_system: str, histogram_name: str) -> str:
    """Generate storage key for session state"""
    return f"{collision_system}_{histogram_name}"


def load_existing_data(collision_system: str, histogram_name: str) -> list[CombinationData]:
    """Load existing data for the current selection"""
    storage_key = get_storage_key(collision_system, histogram_name)

    if storage_key not in st.session_state.combinations_data:
        combinations = generate_combinations()
        data: list[CombinationData] = []

        existing_subset = EXISTING_LOOKUP.get(collision_system, {}).get(histogram_name, {})

        for combo in combinations:
            combo_key = json.dumps(combo, sort_keys=True)
            existing = existing_subset.get(combo_key, {})

            data.append(
                {
                    "parameters": combo,
                    "table": existing.get("table"),
                    "entry": existing.get("entry"),
                    "systematics": {},  # str -> str
                    "additional_systematics": {},  # str -> float
                }
            )

        st.session_state.combinations_data[storage_key] = data

    return st.session_state.combinations_data[storage_key]


def save_data(collision_system: str, histogram_name: str, data: list[CombinationData]) -> None:
    """Save data to session state"""
    storage_key = get_storage_key(collision_system, histogram_name)
    st.session_state.combinations_data[storage_key] = data


def format_params_badges(params: ParameterDict) -> str:
    """Format parameters as colored badges with better theme compatibility"""
    badges = []
    # Use more muted colors that work better with both light and dark themes
    colors = ["#FF8A80", "#80CBC4", "#81C784", "#FFB74D"]  # Softer colors

    for i, param in enumerate(PARAM_ORDER):
        value = params[param]
        if isinstance(value, list):
            value_str = f"[{', '.join(map(str, value))}]"
        else:
            value_str = str(value)

        color = colors[i % len(colors)]
        badges.append(f"""
            <span style="
                background-color: {color};
                color: #1e1e1e;
                padding: 3px 8px;
                margin: 2px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: 500;
                display: inline-block;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            ">
                {param}: {value_str}
            </span>
        """)

    return "".join(badges)


def create_combination_summary(combo_data: CombinationData, index: int) -> tuple[str, str]:
    """Create a summary for the combination with status indicators"""
    params = combo_data["parameters"]

    # Status indicators
    status_parts = []
    if combo_data["table"] is not None and combo_data["entry"] is not None:
        status_parts.append("📊 Table/Entry")
    if combo_data["systematics"]:
        status_parts.append(f"🔧 {len(combo_data['systematics'])} Sys")
    if combo_data["additional_systematics"]:
        status_parts.append(f"➕ {len(combo_data['additional_systematics'])} Add'l")

    status = " | ".join(status_parts) if status_parts else "⚠️ Incomplete"

    # Create title with index and status
    title = f"**Combination {index + 1}** - {status}"

    # Create parameter display
    param_display = format_params_badges(params)

    return title, param_display


def create_combinations_grid_view(current_data: list[CombinationData]) -> None:
    """Create a grid view of combinations for quick overview"""
    st.subheader("Quick Grid View")

    # Create grid with 2 combinations per row
    cols_per_row = 2
    for i in range(0, len(current_data), cols_per_row):
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            idx = i + j
            if idx >= len(current_data):
                break

            combo_data = current_data[idx]

            with cols[j]:
                # Use Streamlit container with custom styling that respects theme
                with st.container(border=True):
                    st.markdown(f"**Combination {idx + 1}**")

                    # Display parameters using badges
                    st.markdown(format_params_badges(combo_data["parameters"]), unsafe_allow_html=True)

                    # Status information using columns for better layout
                    status_col1, status_col2 = st.columns(2)

                    with status_col1:
                        table_status = combo_data["table"] if combo_data["table"] is not None else "Not set"
                        entry_status = combo_data["entry"] if combo_data["entry"] is not None else "Not set"
                        st.caption(f"**Table:** {table_status}")
                        st.caption(f"**Entry:** {entry_status}")

                    with status_col2:
                        sys_count = len(combo_data["systematics"])
                        add_sys_count = len(combo_data["additional_systematics"])
                        st.caption(f"**Systematics:** {sys_count}")
                        st.caption(f"**Additional:** {add_sys_count}")

                    # Status indicator with emoji
                    if combo_data["table"] is not None and combo_data["entry"] is not None:
                        if sys_count > 0 or add_sys_count > 0:
                            st.success("✅ Complete")
                        else:
                            st.info("📊 Basic setup done")
                    else:
                        st.warning("⚠️ Incomplete")

                    # Quick edit button
                    if st.button("Edit Details", key=f"quick_edit_{idx}", use_container_width=True):
                        st.session_state[f"expand_combo_{idx}"] = True
                        st.rerun()


def edit_dict_interface(dict_data: dict[str, Any], key_prefix: str, value_type: str = "str") -> None:
    """Create an interface for editing dictionary data"""
    st.write(f"**{key_prefix.replace('_', ' ').title()}:**")

    # Display existing entries
    to_remove: list[str] = []
    for i, (key, value) in enumerate(dict_data.items()):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_key = st.text_input("Key", value=key, key=f"{key_prefix}_key_{i}")

        with col2:
            if value_type == "float":
                new_value = st.number_input(
                    "Value", value=float(value) if value else 0.0, key=f"{key_prefix}_val_{i}", format="%.6f"
                )
            else:
                new_value = st.text_input("Value", value=str(value), key=f"{key_prefix}_val_{i}")

        with col3:
            if st.button("🗑️", key=f"{key_prefix}_del_{i}", help="Delete this entry"):
                to_remove.append(key)

        # Update the dictionary
        if new_key != key:
            dict_data.pop(key, None)
            if new_key:
                dict_data[new_key] = new_value
        elif new_key:
            dict_data[new_key] = new_value

    # Remove marked entries
    for key in to_remove:
        dict_data.pop(key, None)

    # Add new entry interface
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        new_key = st.text_input("New key", key=f"{key_prefix}_new_key")

    with col2:
        if value_type == "float":
            new_value = st.number_input("New value", value=0.0, key=f"{key_prefix}_new_val", format="%.6f")
        else:
            new_value = st.text_input("New value", key=f"{key_prefix}_new_val")

    with col3:
        if st.button("➕", key=f"{key_prefix}_add", help="Add new entry"):
            if new_key and new_value:
                dict_data[new_key] = new_value
                st.rerun()


def build_nested_yaml(combos: list[CombinationData], level: int = 0) -> list[dict[str, Any]]:
    """Build nested YAML structure including systematics"""
    if level >= len(PARAM_ORDER):
        return []

    param = PARAM_ORDER[level]
    grouped: dict[str, list[CombinationData]] = {}

    for combo in combos:
        val_key = json.dumps(combo["parameters"][param])
        if val_key not in grouped:
            grouped[val_key] = []
        grouped[val_key].append(combo)

    result: list[dict[str, Any]] = []
    for val_str, group in grouped.items():
        val = json.loads(val_str)
        group_entry: dict[str, Any] = {"parameters": {param: [val] if not isinstance(val, list) else [val]}}

        if level == len(PARAM_ORDER) - 1:
            if len(group) == 1:
                item = group[0]
                group_entry["table"] = item["table"]
                group_entry["entry"] = item["entry"]
                if item["systematics"]:
                    group_entry["systematics"] = item["systematics"]
                if item["additional_systematics"]:
                    group_entry["additional_systematics"] = item["additional_systematics"]
            else:
                group_entry["combinations"] = []
                for item in group:
                    combo_entry: dict[str, Any] = {"parameters": {}, "table": item["table"], "entry": item["entry"]}
                    if item["systematics"]:
                        combo_entry["systematics"] = item["systematics"]
                    if item["additional_systematics"]:
                        combo_entry["additional_systematics"] = item["additional_systematics"]
                    group_entry["combinations"].append(combo_entry)
        else:
            group_entry["combinations"] = build_nested_yaml(group, level + 1)

        result.append(group_entry)

    return result


def apply_bulk_changes(
    current_data: list[CombinationData],
    bulk_selections: dict[str, list[Any]],
    bulk_table: int | None,
    bulk_entry: int | None,
) -> None:
    """Apply bulk changes to selected combinations"""
    for combo_data in current_data:
        params = combo_data["parameters"]
        match = True

        for param, selected_vals in bulk_selections.items():
            if selected_vals and params[param] not in selected_vals:
                match = False
                break

        if match:
            if bulk_table is not None:
                combo_data["table"] = bulk_table
            if bulk_entry is not None:
                combo_data["entry"] = bulk_entry


def create_overview_dataframe(current_data: list[CombinationData]) -> pd.DataFrame:
    """Create overview dataframe for display"""
    overview_data: list[dict[str, Any]] = []
    for combo_data in current_data:
        row: dict[str, Any] = {}
        for param in PARAM_ORDER:
            row[param] = str(combo_data["parameters"][param])
        row["Table"] = combo_data["table"]
        row["Entry"] = combo_data["entry"]
        row["Systematics Count"] = len(combo_data["systematics"])
        row["Additional Sys Count"] = len(combo_data["additional_systematics"])
        overview_data.append(row)

    return pd.DataFrame(overview_data)


def setup_observables() -> None:
    if not st.session_state.observables:
        _here = Path(__file__).parent
        st.session_state.observables = observable.read_observables_from_config(
            jetscape_analysis_config_path=_here.parent.parent / "config"
        )


def main() -> None:
    """Main Streamlit application"""

    # Setup
    setup_observables()

    st.title("Configure data sources")

    observable = st.selectbox(
        "Observable: sqrt_s, exp., observable class, observable name (internal name)",
        st.session_state.observables.values(),
        format_func=lambda o: f"{o.sqrt_s}, {o.experiment}, {o.observable_class}, {o.display_name} ({o.internal_name_without_experiment})",
    )

    # Configure the histogram
    col1, col2 = st.columns(2)

    with col1:
        collision_system = st.selectbox("Collision System", ["AA", "pp"])

    with col2:
        histogram_options = ["spectra", "ratio"] if collision_system == "AA" else ["spectra"]
        histogram_name = st.selectbox("Histogram Name", histogram_options, accept_new_options=True)

    st.write(f'Configuring histogram "{histogram_name}"')

    st.divider()

    # Now that we know what we're working with, we need to setup the parameters
    all_parameters = observable.parameters()

    # Load current data
    # TODO(RJE): Need to update this parsing...
    current_data = load_existing_data(collision_system, histogram_name)

    # Bulk edit section
    st.header("Bulk Edit Selection")

    bulk_selections: dict[str, list[Any]] = {}
    bulk_cols = st.columns(len(all_parameters))

    for i, param in enumerate(all_parameters):
        with bulk_cols[i]:
            param_name = param.encode_name()
            param_display_name = " ".join(param.encode_name().split("_"))
            st.subheader(param_display_name)
            select_all = st.checkbox("Select all", key=f"select_all_{param_name}")

            selected_values: list[Any] = []
            for value in param.values:
                value_str = str(value)
                is_selected = st.checkbox(value_str, key=f"{param_name}_{value_str}", value=select_all)
                if is_selected:
                    selected_values.append(value)

            bulk_selections[param_name] = selected_values

    # Bulk edit inputs
    bulk_col1, bulk_col2, bulk_col3 = st.columns([1, 1, 2])

    with bulk_col1:
        bulk_table = st.number_input("HEPdata Table identifier", value=None, step=1, key="bulk_table")

    with bulk_col2:
        bulk_entry = st.number_input("HEPData table index", value=None, step=1, key="bulk_entry")

    with bulk_col3:
        if st.button("Apply to Selection"):
            apply_bulk_changes(current_data, bulk_selections, bulk_table, bulk_entry)
            save_data(collision_system, histogram_name, current_data)
            st.rerun()

    if st.button("Reset to Existing Data"):
        storage_key = get_storage_key(collision_system, histogram_name)
        if storage_key in st.session_state.combinations_data:
            del st.session_state.combinations_data[storage_key]
        st.rerun()

    st.divider()

    # Add grid view
    create_combinations_grid_view(current_data)

    st.divider()

    # All combinations - detailed editing (improved)
    st.header("Detailed Configuration")

    # Add search/filter option
    search_term = st.text_input("🔍 Search combinations (parameter values)", placeholder="e.g., 0.2, [0,10], z_cut")

    # Filter combinations based on search
    filtered_indices = []
    for i, combo_data in enumerate(current_data):
        if not search_term:
            filtered_indices.append(i)
        else:
            # Search in parameter values
            param_str = json.dumps(combo_data["parameters"], sort_keys=True).lower()
            if search_term.lower() in param_str:
                filtered_indices.append(i)

    if search_term and not filtered_indices:
        st.warning(f"No combinations found matching '{search_term}'")

    # Show filtered combinations
    for i in filtered_indices:
        combo_data = current_data[i]
        title, param_display = create_combination_summary(combo_data, i)

        # Check if this combination should be expanded
        expand_key = f"expand_combo_{i}"
        is_expanded = st.session_state.get(expand_key, False)

        with st.expander(f"{title}", expanded=is_expanded):
            # Clear the expansion state after showing
            if expand_key in st.session_state:
                del st.session_state[expand_key]

            # Show parameters in a nice format
            st.markdown("**Parameters:**")
            st.markdown(param_display, unsafe_allow_html=True)

            st.markdown("---")

            # Basic info in columns
            col1, col2 = st.columns(2)

            with col1:
                table_val = st.number_input(
                    "Table",
                    value=combo_data["table"] if combo_data["table"] is not None else 0,
                    step=1,
                    key=f"table_{i}",
                    help="HEPData table number",
                )
                combo_data["table"] = table_val if table_val != 0 else None

            with col2:
                entry_val = st.number_input(
                    "Entry",
                    value=combo_data["entry"] if combo_data["entry"] is not None else 0,
                    step=1,
                    key=f"entry_{i}",
                    help="Entry index within the table",
                )
                combo_data["entry"] = entry_val if entry_val != 0 else None

            # Load systematics button with better styling
            if combo_data["table"] is not None and combo_data["entry"] is not None:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("📥 Load Systematics", key=f"load_sys_{i}"):
                        try:
                            filename = Path(f"{collision_system}_data.yaml")
                            loaded_sys = load_hepdata_file(filename, combo_data["table"], combo_data["entry"])
                            combo_data["systematics"].update(loaded_sys)
                            st.success(f"✅ Loaded {len(loaded_sys)} systematic uncertainties")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to load systematics: {e}")
                with col2:
                    st.caption("Load systematic uncertainties from HEPData file using table and entry numbers")
            else:
                st.info("💡 Set both Table and Entry numbers to enable systematic loading")

            st.markdown("---")

            # Systematics editing with better headers
            st.markdown("### 🔧 Systematics (String Values)")
            if combo_data["systematics"]:
                st.caption(f"Currently has {len(combo_data['systematics'])} systematic uncertainties")
            edit_dict_interface(combo_data["systematics"], f"systematics_{i}", "str")

            st.markdown("---")

            # Additional systematics editing
            st.markdown("### ➕ Additional Systematics (Numeric Values)")
            if combo_data["additional_systematics"]:
                st.caption(f"Currently has {len(combo_data['additional_systematics'])} additional systematics")
            edit_dict_interface(combo_data["additional_systematics"], f"additional_systematics_{i}", "float")

    # Save data
    save_data(collision_system, histogram_name, current_data)

    st.divider()

    # Quick overview table (read-only)
    st.header("Quick Overview")
    overview_df = create_overview_dataframe(current_data)
    st.dataframe(overview_df, use_container_width=True)

    st.divider()

    # Export section
    st.header("Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate YAML"):
            yaml_data = {"data": {}}

            for storage_key, combo_data in st.session_state.combinations_data.items():
                cs, hist = storage_key.split("_", 1)

                if cs not in yaml_data["data"]:
                    yaml_data["data"][cs] = {"hepdata": {"filename": f"{cs}_data.yaml"}}

                yaml_data["data"][cs]["hepdata"][hist] = build_nested_yaml(combo_data, 0)

            yaml_str = yaml.dump(yaml_data, default_flow_style=False, allow_unicode=True)
            st.session_state.yaml_output = yaml_str

    with col2:
        if "yaml_output" in st.session_state:
            st.download_button(
                label="Download YAML",
                data=st.session_state.yaml_output,
                file_name="histogram_config.yaml",
                mime="text/yaml",
            )

    # Display YAML output
    if "yaml_output" in st.session_state:
        st.subheader("YAML Output")
        st.code(st.session_state.yaml_output, language="yaml")


if __name__ == "__main__":
    main()
