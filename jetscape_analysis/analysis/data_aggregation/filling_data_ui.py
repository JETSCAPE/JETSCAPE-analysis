"""Generate the UI used to fill the HEPdata YAML configuration.

Since I'm not an expert in HTML + JS, it was created using an LLM.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from jetscape_analysis.base import helpers

logger = logging.getLogger(__name__)


# TODO(RJE): Rename "histogram_name" to something more appropriate

def generate_histogram_config_page(
    expected_grid: dict[str, list[Any]],
    existing_data: list[dict[str, Any]] | None = None,
    title: str = "Histogram Configurator"
) -> str:
    """Generate a standalone HTML page for configuring histogram mappings with nested YAML output.

    Features:
        - Live updates combinations when checkboxes are changed.
        - Supports multiple collision systems and histogram names.

    Args:
        expected_grid: Parameter grid as a dict of parameter -> list of values.
        existing_data: Optional list of existing mappings, where each mapping is:
                       {
                           "collision_system": str,
                           "histogram_name": str,
                           "parameters": dict[str, Any],
                           "table": int,
                           "entry": int
                       }. Default: None.
        title: Title of the HTML page. Default: "Histogram Configurator".

    Returns:
        HTML content as a string.
    """
    param_order = list(expected_grid.keys())

        # Generate full Cartesian product of parameters
    def normalize(v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v

    normalized_grid = {
        k: [normalize(x) for x in v] for k, v in expected_grid.items()
    }
    all_combos = []
    for values in itertools.product(*normalized_grid.values()):
        params = dict(zip(param_order, values, strict=True))
        all_combos.append(params)

    # Ensure consistent key ordering for JSON serialization
    sorted_combos = [{k: combo[k] for k in param_order} for combo in all_combos]

    # Map existing data to quick lookup
    existing_lookup: dict[str, dict[str, Any]] = {}
    if existing_data:
        for entry in existing_data:
            key = json.dumps({k: entry["parameters"][k] for k in param_order})
            existing_lookup[key] = {"table": entry["table"], "entry": entry["entry"]}

    template_str = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    .param-group { margin-bottom: 15px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
    .param-group h3 { margin-top: 0; }
    table { border-collapse: collapse; margin-top: 10px; width: 100%; font-size: 14px; }
    th, td { border: 1px solid #ccc; padding: 5px; text-align: left; }
    .table-entry-input { width: 60px; }
    button { margin: 5px; padding: 6px 10px; }
    .output { white-space: pre; background: #f9f9f9; padding: 10px; border: 1px solid #ccc; }
</style>
</head>
<body>
<h1>{{ title }}</h1>

<!-- Collision + Histogram selection -->
<label>Collision System:
<select id="collision-system">
    <option value="AA">AA</option>
    <option value="pp">pp</option>
</select>
</label>

<label>Histogram Name:
<select id="histogram-name">
    <option value="spectra">spectra</option>
    <option value="ratio">ratio</option>
</select>
<input type="text" id="custom-histogram" placeholder="Custom name">
<button id="set-custom-histogram">Set Custom</button>
</label>

<hr>

<!-- Bulk edit selection -->
<h2>Bulk Edit Selection</h2>
<div id="parameter-selectors">
    {% for param, values in expected_grid.items() %}
    <div class="param-group" data-param="{{ param }}">
        <h3>{{ param }}</h3>
        {% for val in values %}
            <label>
                <input type="checkbox" class="param-value" value="{{ val|tojson }}">
                {{ val }}
            </label><br>
        {% endfor %}
    </div>
    {% endfor %}
</div>
<label>Table: <input type="number" id="bulk-table" class="table-entry-input"></label>
<label>Entry: <input type="number" id="bulk-entry" class="table-entry-input"></label>
<button id="apply-bulk">Apply to Selection</button>

<hr>

<!-- Combinations table -->
<h2>All Combinations</h2>
<table id="combinations-table">
    <thead>
        <tr>
            {% for param in param_order %}
            <th>{{ param }}</th>
            {% endfor %}
            <th>Table</th>
            <th>Entry</th>
        </tr>
    </thead>
    <tbody>
        {% for combo in sorted_combos %}
        {% set key = combo|tojson %}
        <tr data-params='{{ key|e }}'>
            {% for param in param_order %}
            <td>{{ combo[param] }}</td>
            {% endfor %}
            <td><input type="number" class="table-entry-input table-input" value="{{ existing_lookup.get(key, {}).get('table','') }}"></td>
            <td><input type="number" class="table-entry-input entry-input" value="{{ existing_lookup.get(key, {}).get('entry','') }}"></td>
        </tr>
        {% endfor %}
    </tbody>
</table>

<hr>
<button id="save-local">Save to Local Storage</button>
<button id="load-local">Load from Local Storage</button>
<button id="export-yaml">Export YAML</button>

<h2>YAML Output</h2>
<pre id="yaml-output" class="output"></pre>

<script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
<script>
const paramOrder = {{ param_order|tojson }};

document.getElementById('set-custom-histogram').addEventListener('click', () => {
    const customName = document.getElementById('custom-histogram').value.trim();
    if (customName) {
        const histSelect = document.getElementById('histogram-name');
        let opt = document.createElement('option');
        opt.value = customName;
        opt.textContent = customName;
        histSelect.appendChild(opt);
        histSelect.value = customName;
    }
});

document.getElementById('apply-bulk').addEventListener('click', () => {
    const selectedParams = {};
    document.querySelectorAll('.param-group').forEach(group => {
        const param = group.dataset.param;
        const checked = Array.from(group.querySelectorAll('.param-value:checked')).map(cb => JSON.parse(cb.value));
        if (checked.length > 0) selectedParams[param] = checked;
    });

    const bulkTable = document.getElementById('bulk-table').value;
    const bulkEntry = document.getElementById('bulk-entry').value;

    document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
        const params = JSON.parse(tr.dataset.params);
        let match = true;
        for (let key in selectedParams) {
            if (!selectedParams[key].some(v => JSON.stringify(v) === JSON.stringify(params[key]))) {
                match = false;
                break;
            }
        }
        if (match) {
            if (bulkTable !== '') tr.querySelector('.table-input').value = bulkTable;
            if (bulkEntry !== '') tr.querySelector('.entry-input').value = bulkEntry;
        }
    });
});

document.getElementById('save-local').addEventListener('click', () => {
    const cs = document.getElementById('collision-system').value;
    const hist = document.getElementById('histogram-name').value;
    const key = 'histogramConfig_' + cs + '_' + hist;
    const combos = [];
    document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
        combos.push({
            parameters: JSON.parse(tr.dataset.params),
            table: parseInt(tr.querySelector('.table-input').value) || null,
            entry: parseInt(tr.querySelector('.entry-input').value) || null
        });
    });
    localStorage.setItem(key, JSON.stringify(combos));
    alert('Configuration saved for ' + cs + ' / ' + hist);
});

document.getElementById('load-local').addEventListener('click', () => {
    const cs = document.getElementById('collision-system').value;
    const hist = document.getElementById('histogram-name').value;
    const key = 'histogramConfig_' + cs + '_' + hist;
    const stored = localStorage.getItem(key);
    if (stored) {
        const combos = JSON.parse(stored);
        const lookup = {};
        combos.forEach(c => {
            lookup[JSON.stringify(c.parameters, Object.keys(c.parameters).sort())] = c;
        });
        document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
            const paramsKey = JSON.stringify(JSON.parse(tr.dataset.params), Object.keys(JSON.parse(tr.dataset.params)).sort());
            if (lookup[paramsKey]) {
                tr.querySelector('.table-input').value = lookup[paramsKey].table ?? '';
                tr.querySelector('.entry-input').value = lookup[paramsKey].entry ?? '';
            }
        });
        alert('Configuration loaded for ' + cs + ' / ' + hist);
    } else {
        alert('No saved configuration found for ' + cs + ' / ' + hist);
    }
});

function buildNestedYAML(combos, level=0) {
    if (level >= paramOrder.length) return [];

    const param = paramOrder[level];
    const grouped = {};
    combos.forEach(c => {
        const val = JSON.stringify(c.parameters[param]);
        if (!grouped[val]) grouped[val] = [];
        grouped[val].push(c);
    });

    const result = [];
    for (let valStr in grouped) {
        const val = JSON.parse(valStr);
        const groupEntry = { parameters: { [param]: Array.isArray(val) ? [val] : [val] } };

        if (level === paramOrder.length - 1) {
            const leaves = grouped[valStr];
            if (leaves.length === 1) {
                groupEntry.table = leaves[0].table;
                groupEntry.entry = leaves[0].entry;
            } else {
                groupEntry.combinations = leaves.map(l => ({
                    parameters: {},
                    table: l.table,
                    entry: l.entry
                }));
            }
        } else {
            groupEntry.combinations = buildNestedYAML(grouped[valStr], level + 1);
        }
        result.push(groupEntry);
    }
    return result;
}

document.getElementById('export-yaml').addEventListener('click', () => {
    const yamlObj = { data: {} };

    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key.startsWith('histogramConfig_')) {
            const parts = key.replace('histogramConfig_', '').split('_');
            const cs = parts[0];
            const hist = parts.slice(1).join('_');
            const combos = JSON.parse(localStorage.getItem(key));
            yamlObj.data[cs] = yamlObj.data[cs] || { hepdata: { filename: cs + "_data.yaml" } };
            yamlObj.data[cs].hepdata[hist] = buildNestedYAML(combos, 0);
        }
    }

    const yamlStr = jsyaml.dump(yamlObj, { noRefs: true });
    document.getElementById('yaml-output').textContent = yamlStr;
});
</script>
</body>
</html>
"""
    template = Template(template_str)
    return template.render(
        title=title,
        expected_grid=expected_grid,
        param_order=param_order,
        sorted_combos=sorted_combos,
        existing_lookup=existing_lookup
    )


def main() -> None:
    helpers.setup_logging(level=logging.INFO)

    expected_grid = {
        "jet_R": [0.2, 0.3],
        "soft_drop": ["z_cut_02_beta_0"],
        "centrality": [[0, 10], [10, 20]],
        "jet_pt": [[100, 200], [200, 300]],
    }
    existing_data = [
        {"collision_system": "AA", "histogram_name": "spectra", "parameters": {"jet_R": 0.2, "soft_drop": "z_cut_02_beta_0", "centrality": [0, 10], "jet_pt": [100, 200]}, "table": 15, "entry": 0},
        {"collision_system": "pp", "histogram_name": "spectra", "parameters": {"jet_R": 0.3, "soft_drop": "z_cut_02_beta_0", "centrality": [10, 20], "jet_pt": [200, 300]}, "table": 12, "entry": 1}
    ]
    html_content = generate_histogram_config_page(expected_grid, existing_data)

    with Path("histogram_config.html").open("w") as f:
        f.write(html_content)

    logger.info("HTML page written to histogram_config.html")

if __name__ == "__main__":
    main()