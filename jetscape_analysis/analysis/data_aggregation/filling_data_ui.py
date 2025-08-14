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
    title: str = "Histogram Configurator",
) -> str:
    """
    HTML page for editing histogram mappings.
    Shows full parameter grid, auto-saves to localStorage, supports reset, copy-to-clipboard,
    and outputs YAML with inline parameter lists only.
    """
    param_order = list(expected_grid.keys())

    # Generate full Cartesian product
    all_combos = []
    for values in itertools.product(*expected_grid.values()):
        params = dict(zip(param_order, values, strict=True))
        all_combos.append(params)

    # Precompute JSON keys consistently
    def json_key(params: dict[str, Any]) -> str:
        return json.dumps(params, separators=(",", ":"), sort_keys=True)

    sorted_combos = []
    for combo in all_combos:
        sorted_combo = {k: combo[k] for k in param_order}
        sorted_combos.append({"params": sorted_combo, "json_key": json_key(sorted_combo)})

    # Map existing data for quick lookup, nested by (collision_system, histogram_name)
    existing_lookup: dict[str, dict[str, dict[str, Any]]] = {}
    if existing_data:
        for entry in existing_data:
            cs = entry.get("collision_system", "AA")
            hist = entry.get("histogram_name", "spectra")
            key = json_key({k: entry["parameters"][k] for k in param_order})
            existing_lookup.setdefault(cs, {}).setdefault(hist, {})[key] = {
                "table": entry["table"],
                "entry": entry["entry"],
            }

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

<h2>Bulk Edit Selection</h2>
<div id="parameter-selectors">
    {% for param, values in expected_grid.items() %}
    <div class="param-group" data-param="{{ param }}">
        <h3>{{ param }}</h3>
        <label><input type="checkbox" class="select-all"> <strong>Select All</strong></label><br>
        {% for val in values %}
            <label>
                <input type="checkbox" class="param-value" value="{{ (val if val is not none else 'null')|tojson|e }}">
                {{ val }}
            </label><br>
        {% endfor %}
    </div>
    {% endfor %}
</div>
<label>Table: <input type="number" id="bulk-table" class="table-entry-input"></label>
<label>Entry: <input type="number" id="bulk-entry" class="table-entry-input"></label>
<button id="apply-bulk">Apply to Selection</button>
<button id="reset-storage">Reset to Existing Data</button>

<hr>

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
        <tr data-params='{{ combo.json_key }}'>
            {% for param in param_order %}
            <td>{{ combo.params[param] }}</td>
            {% endfor %}
            <td><input type="number" class="table-entry-input table-input"></td>
            <td><input type="number" class="table-entry-input entry-input"></td>
        </tr>
        {% endfor %}
    </tbody>
</table>

<hr>
<button id="export-yaml">Export YAML</button>
<button id="copy-yaml">Copy to Clipboard</button>

<h2>YAML Output</h2>
<pre id="yaml-output" class="output"></pre>

<script src="https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js"></script>
<script>
const paramOrder = {{ param_order|tojson }};
const existingLookup = {{ existing_lookup|tojson }};

function storageKey(cs, hist) {
    return 'histogramConfig_' + cs + '_' + hist;
}

function clearBulkSelections() {
    document.querySelectorAll('.param-value').forEach(cb => cb.checked = false);
    document.querySelectorAll('.select-all').forEach(cb => cb.checked = false);
    document.getElementById('bulk-table').value = '';
    document.getElementById('bulk-entry').value = '';
}

function loadDataToTable() {
    clearBulkSelections();
    const cs = document.getElementById('collision-system').value;
    const hist = document.getElementById('histogram-name').value;
    const key = storageKey(cs, hist);
    let combos = JSON.parse(localStorage.getItem(key) || 'null');

    if (!combos) {
        combos = [];
        const existingSubset = (existingLookup[cs] && existingLookup[cs][hist]) ? existingLookup[cs][hist] : {};
        document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
            const paramsKey = tr.dataset.params;
            const existing = existingSubset[paramsKey] || {};
            combos.push({
                parameters: JSON.parse(paramsKey),
                table: existing.table ?? null,
                entry: existing.entry ?? null
            });
        });
        localStorage.setItem(key, JSON.stringify(combos));
    }

    const lookup = {};
    combos.forEach(c => lookup[JSON.stringify(c.parameters, Object.keys(c.parameters).sort())] = c);
    document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
        const paramsKey = tr.dataset.params;
        if (lookup[paramsKey]) {
            tr.querySelector('.table-input').value = lookup[paramsKey].table ?? '';
            tr.querySelector('.entry-input').value = lookup[paramsKey].entry ?? '';
        } else {
            tr.querySelector('.table-input').value = '';
            tr.querySelector('.entry-input').value = '';
        }
    });
}

function saveTableToStorage() {
    const cs = document.getElementById('collision-system').value;
    const hist = document.getElementById('histogram-name').value;
    const key = storageKey(cs, hist);
    const combos = [];
    document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
        combos.push({
            parameters: JSON.parse(tr.dataset.params),
            table: parseInt(tr.querySelector('.table-input').value) || null,
            entry: parseInt(tr.querySelector('.entry-input').value) || null
        });
    });
    localStorage.setItem(key, JSON.stringify(combos));
}

document.getElementById('collision-system').addEventListener('change', loadDataToTable);
document.getElementById('histogram-name').addEventListener('change', loadDataToTable);

document.querySelectorAll('.select-all').forEach(cb => {
    cb.addEventListener('change', function() {
        const group = this.closest('.param-group');
        group.querySelectorAll('.param-value').forEach(valCb => valCb.checked = this.checked);
    });
});

function valuesEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

document.getElementById('apply-bulk').addEventListener('click', () => {
    const selectedParams = {};
    document.querySelectorAll('.param-group').forEach(group => {
        const param = group.dataset.param;
        const checked = [];
        Array.from(group.querySelectorAll('.param-value:checked')).forEach(cb => {
            if (!cb.value || cb.value.trim() === '') return; // skip empty
            try {
                checked.push(JSON.parse(cb.value));
            } catch (e) {
                console.warn("Invalid JSON in checkbox value:", cb.value, e);
            }
        });
        if (checked.length > 0) selectedParams[param] = checked;
    });

    const bulkTable = document.getElementById('bulk-table').value;
    const bulkEntry = document.getElementById('bulk-entry').value;

    document.querySelectorAll('#combinations-table tbody tr').forEach(tr => {
        const params = JSON.parse(tr.dataset.params);
        let match = true;
        for (let key in selectedParams) {
            if (!selectedParams[key].some(v => valuesEqual(v, params[key]))) {
                match = false;
                break;
            }
        }
        if (match) {
            if (bulkTable !== '') tr.querySelector('.table-input').value = bulkTable;
            if (bulkEntry !== '') tr.querySelector('.entry-input').value = bulkEntry;
        }
    });

    saveTableToStorage();
    clearBulkSelections();
});

document.querySelector('#combinations-table').addEventListener('input', saveTableToStorage);

document.getElementById('reset-storage').addEventListener('click', () => {
    const cs = document.getElementById('collision-system').value;
    const hist = document.getElementById('histogram-name').value;
    const key = storageKey(cs, hist);
    localStorage.removeItem(key);
    loadDataToTable();
    alert('Reset to existing data for ' + cs + ' / ' + hist);
});

function markParameterListsInline(obj) {
    if (Array.isArray(obj)) {
        return obj;
    } else if (obj && typeof obj === 'object') {
        const newObj = {};
        for (let k in obj) {
            if (k === 'parameters' && typeof obj[k] === 'object') {
                const paramsObj = {};
                for (let pk in obj[k]) {
                    if (Array.isArray(obj[k][pk])) {
                        paramsObj[pk] = Object.assign([], obj[k][pk]);
                        paramsObj[pk].style = 'flow'; // mark for inline
                    } else {
                        paramsObj[pk] = obj[k][pk];
                    }
                }
                newObj[k] = paramsObj;
            } else {
                newObj[k] = markParameterListsInline(obj[k]);
            }
        }
        return newObj;
    }
    return obj;
}

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

    const processed = markParameterListsInline(yamlObj);
    const yamlStr = jsyaml.dump(processed, { noRefs: true, styles: { '!!seq': 'flow' } });
    document.getElementById('yaml-output').textContent = yamlStr;
});

document.getElementById('copy-yaml').addEventListener('click', () => {
    const yamlText = document.getElementById('yaml-output').textContent;
    navigator.clipboard.writeText(yamlText).then(() => {
        alert('YAML copied to clipboard');
    });
});

loadDataToTable();
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
        existing_lookup=existing_lookup,
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
        {
            "collision_system": "AA",
            "histogram_name": "spectra",
            "parameters": {"jet_R": 0.2, "soft_drop": "z_cut_02_beta_0", "centrality": [0, 10], "jet_pt": [100, 200]},
            "table": 15,
            "entry": 0,
        },
        {
            "collision_system": "pp",
            "histogram_name": "spectra",
            "parameters": {"jet_R": 0.3, "soft_drop": "z_cut_02_beta_0", "centrality": [10, 20], "jet_pt": [200, 300]},
            "table": 12,
            "entry": 1,
        },
    ]
    html_content = generate_histogram_config_page(expected_grid, existing_data)

    with Path("histogram_config.html").open("w") as f:
        f.write(html_content)

    logger.info("HTML page written to histogram_config.html")


if __name__ == "__main__":
    main()
