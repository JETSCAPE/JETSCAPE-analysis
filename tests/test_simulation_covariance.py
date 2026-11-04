"""Synthetic event-to-Bayesian-bank checks; no production data or reconstruction."""

import ast
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jetscape_analysis.analysis import simulation_covariance as m


def test_same_event_jets_are_not_independent():
    moments = m.EventMoments(4)
    vectors = np.array([[6, 3, 0, 3], [0, 0, 2, 2], [0, 0, 0, 4]])
    for vector in vectors:
        moments.update(vector)
    np.testing.assert_array_equal(moments.total, vectors.sum(0))
    np.testing.assert_array_equal(moments.outer, vectors.T @ vectors)
    assert moments.outer[0, 0] == 36  # two jets in one event, not two independent events
    assert moments.outer[0, 1] == 18


def test_histogram_edges_and_shape_counter():
    counts, norm = m.histogram_counts([-1, 0, 0.1, 0.999, 1, 2], [0, 0.1, 1])
    np.testing.assert_array_equal(counts, [1, 2])
    assert norm == 4


def rows(values, offset=0):
    return [
        {
            "event_ID": i,
            "pt_hat": float(10 + i),
            "vertex_x": float(i),
            "vertex_y": -float(i),
            "event_weight_all": float(i + 1),
            "x": v,
            "event_weight": None,
        }
        for i, v in enumerate(values, offset)
    ]


def write_stream(path, values):
    pq.write_table(pa.Table.from_pylist(values), path)
    return str(path)


BLOCK = {"name": "x", "columns": {"x": 1.0}, "edges": [0.0, 1.0, 2.0]}


def collect_case(tmp_path, name="aa", paired=True):
    first = rows([[0.2, 0.3, 1.2], [], [1.1], [0.4]])
    if name == "pp":
        for row in first:
            row["event_ID"] += 1000
    a = write_stream(tmp_path / f"{name}_0.parquet", first)
    streams = [{"name": "d0", "file": a, "blocks": [deepcopy(BLOCK)]}]
    if paired:
        b = write_stream(tmp_path / f"{name}_1.parquet", rows([[0.5, 1.5], [0.8], [], [1.8]]))
        streams.append({"name": "d1", "file": b, "blocks": [deepcopy(BLOCK)]})
    manifest = {"cohort": name, "identity_columns": ["event_ID", "pt_hat", "vertex_x", "vertex_y"], "streams": streams}
    path = tmp_path / f"{name}.npz"
    m.collect(manifest, path)
    return manifest, path


def test_collect_preserves_paired_design_and_empty_event_moments(tmp_path):
    _, path = collect_case(tmp_path)
    expected = np.array([[2, 1, 1, 1, 1, 1], [0, 0, 2, 2, 0, 2], [0, 3, 3, 0, 0, 3], [4, 0, 4, 0, 4, 4]])
    with np.load(path) as z:
        np.testing.assert_array_equal(z["total"], expected.sum(0))
        np.testing.assert_array_equal(z["outer"], expected.T @ expected)
        assert z["events"] == 4
        assert z["outer"][0, 3] != 0


@pytest.mark.parametrize(
    "problem", ["missing_id", "reordered", "duplicate_id", "wrong_weight", "short", "missing_column"]
)
def test_invalid_event_alignment_fails_without_output(tmp_path, problem):
    values = rows([[0.2], [1.2]])
    second = deepcopy(values)
    if problem == "missing_id":
        for row in second:
            del row["event_ID"]
    elif problem == "reordered":
        second.reverse()
    elif problem == "duplicate_id":
        second[1]["event_ID"] = second[0]["event_ID"]
        values[1]["event_ID"] = values[0]["event_ID"]
    elif problem == "wrong_weight":
        second[0]["event_weight"] = 99.0
    elif problem == "short":
        second.pop()
    else:
        for row in second:
            del row["x"]
    streams = [
        {"name": f"d{i}", "file": write_stream(tmp_path / f"s{i}.parquet", content), "blocks": [BLOCK]}
        for i, content in enumerate((values, second))
    ]
    output = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match=r"Missing|Nonfinite|weights differ|different event counts"):
        m.collect({"cohort": "aa", "streams": streams}, output)
    assert not output.exists()


def test_signed_columns_and_explicit_normalization_counters():
    block = dict(BLOCK, columns={"x": 1.0, "holes": -1.0}, normalization="underflow_and_visible")
    row = {"event_weight_all": 2.0, "x": [-1.0, 0.2, 0.4, 1.2, 2.1], "holes": [0.5]}
    np.testing.assert_array_equal(m._event_vector(row, [block]), [2.0, 2.0, 6.0, 2.0])
    block["normalization"] = "all_entries"
    np.testing.assert_array_equal(m._event_vector(row, [block]), [2.0, 2.0, 8.0, 2.0])


def test_ratio_map_derivatives_and_normalization_anticorrelation():
    labels = ["a", "b"]
    terms = [
        [{"numerator": {"a": 1}, "denominator": {"a": 1, "b": 1}, "scale": 1}],
        [{"numerator": {"b": 1}, "denominator": {"a": 1, "b": 1}, "scale": 1}],
    ]
    total = np.array([40.0, 60.0])
    value, jac = m.spectrum_map(total, labels, terms)
    np.testing.assert_allclose(value, [0.4, 0.6])
    covariance = np.asarray((jac @ np.diag(total)) @ jac.T)
    np.testing.assert_allclose(covariance, [[0.0024, -0.0024], [-0.0024, 0.0024]])
    for i in range(2):
        step = np.eye(2)[i] * 1e-4
        delta = (m.spectrum_map(total + step, labels, terms)[0] - m.spectrum_map(total - step, labels, terms)[0]) / 2e-4
        np.testing.assert_allclose(jac.toarray()[:, i], delta, atol=1e-12)


def term(cohort, stream, index, scale=1.0):
    return [
        {
            "numerator": {f"{cohort}/{stream}:x:bin{index}": 1.0},
            "denominator": {f"{cohort}/{stream}:events": 1.0},
            "scale": scale,
        }
    ]


def export_case(tmp_path):
    _, aa = collect_case(tmp_path)
    _, pp = collect_case(tmp_path, "pp", paired=False)
    return {
        "design_ids": [3, 17],
        "features": [{"name": "toy", "xmin": 0.0, "xmax": 1.0}, {"name": "toy", "xmin": 1.0, "xmax": 2.0}],
        "cohorts": {"aa": [str(aa)], "pp": [str(pp)]},
        "spectra": [[term("aa", f"d{d}", i, scale=2.0) for i in range(2)] for d in range(2)],
        "reference": [term("pp", "d0", i, scale=3.0) for i in range(2)],
    }


def test_end_to_end_export_against_dense_independent_oracle(tmp_path):
    manifest = export_case(tmp_path)
    output = tmp_path / "bank.npz"
    m.export(manifest, output)
    a = np.array([[2, 1, 1, 1, 1, 1], [0, 0, 2, 2, 0, 2], [0, 3, 3, 0, 0, 3], [4, 0, 4, 0, 4, 4]], float)
    p = a[:, :3]
    total = np.r_[a.sum(0), p.sum(0)]
    raw = np.zeros((9, 9))
    raw[:6, :6], raw[6:, 6:] = a.T @ a, p.T @ p

    def independent(x):
        aa = np.array([2 * x[:2] / x[2], 2 * x[3:5] / x[5]])
        pp = 3 * x[6:8] / x[8]
        return (aa / pp).ravel()

    eps = 1e-4
    jac = np.stack(
        [(independent(total + eps * e) - independent(total - eps * e)) / (2 * eps) for e in np.eye(9)], axis=1
    )
    with np.load(output) as z:
        np.testing.assert_allclose(z["central_values"].ravel(), independent(total))
        np.testing.assert_allclose(z["covariance"].reshape(4, 4), jac @ raw @ jac.T, atol=1e-10)
        assert z["covariance"].shape == (2, 2, 2, 2)
        assert np.count_nonzero(z["covariance"][0, :, 1, :])  # paired AA and common pp
        np.testing.assert_array_equal(z["design_ids"], [3, 17])


def test_pooled_and_equal_separately_normalized_components():
    total = np.array([2.0, 4.0, 3.0, 9.0])
    labels = ["a:n", "a:d", "b:n", "b:d"]
    terms = [
        [{"numerator": {"a:n": 1, "b:n": 1}, "denominator": {"a:d": 1, "b:d": 1}, "scale": 2.0}],
        [
            {"numerator": {"a:n": 1}, "denominator": {"a:d": 1}, "scale": 1.0},
            {"numerator": {"b:n": 1}, "denominator": {"b:d": 1}, "scale": 1.0},
        ],
    ]
    values, _ = m.spectrum_map(total, labels, terms)
    np.testing.assert_allclose(values, [2 * 5 / 13, 0.5 + 1 / 3])
    assert values[0] != values[1]


def test_partially_overlapping_observables_use_their_own_event_denominators(tmp_path):
    all_rows = rows([[0.2, 0.3, 1.2], [], [1.1], [0.4]])
    old_path = write_stream(tmp_path / "old_only.parquet", all_rows[:2])
    overlap = deepcopy(all_rows[2:])
    overlap[0]["y"], overlap[1]["y"] = [0.2], []
    overlap_path = write_stream(tmp_path / "overlap.parquet", overlap)
    y_block = {"name": "y", "columns": {"y": 1.0}, "edges": [0.0, 1.0]}
    for name, file, blocks in [("old", old_path, [BLOCK]), ("overlap", overlap_path, [BLOCK, y_block])]:
        m.collect(
            {"cohort": name, "streams": [{"name": "d0", "file": file, "blocks": blocks}]}, tmp_path / f"{name}.npz"
        )
    _, pp_path = collect_case(tmp_path, "pp", paired=False)
    terms = [
        [
            {
                "numerator": {"old/d0:x:bin0": 1.0, "overlap/d0:x:bin0": 1.0},
                "denominator": {"old/d0:events": 1.0, "overlap/d0:events": 1.0},
                "scale": 1.0,
            }
        ],
        [{"numerator": {"overlap/d0:y:bin0": 1.0}, "denominator": {"overlap/d0:events": 1.0}, "scale": 1.0}],
    ]
    manifest = {
        "design_ids": [3],
        "features": [{"name": n, "xmin": 0.0, "xmax": 1.0} for n in ("old", "new")],
        "cohorts": {
            "old": [str(tmp_path / "old.npz")],
            "overlap": [str(tmp_path / "overlap.npz")],
            "pp": [str(pp_path)],
        },
        "spectra": [terms],
        "reference": [term("pp", "d0", 0), term("pp", "d0", 0)],
    }
    output = tmp_path / "partial.npz"
    m.export(manifest, output)
    old = np.array([[2, 1, 1], [0, 0, 2]], float)
    new = np.array([[0, 3, 3, 3], [4, 0, 0, 4]], float)
    pp = np.array([[2, 1, 1], [0, 0, 2], [0, 3, 3], [4, 0, 4]], float)
    total = np.r_[old.sum(0), new.sum(0), pp.sum(0)]
    raw = np.zeros((10, 10))
    raw[:3, :3], raw[3:7, 3:7], raw[7:, 7:] = old.T @ old, new.T @ new, pp.T @ pp

    def oracle(x):
        return np.array([(x[0] + x[3]) / (x[2] + x[6]), x[5] / x[6]]) / (x[7] / x[9])

    eps = 1e-4
    jac = np.stack([(oracle(total + eps * e) - oracle(total - eps * e)) / (2 * eps) for e in np.eye(10)], axis=1)
    with np.load(output) as data:
        np.testing.assert_allclose(data["central_values"][0], oracle(total))
        np.testing.assert_allclose(data["covariance"].reshape(2, 2), jac @ raw @ jac.T, atol=1e-10)


def test_batch_accumulation_preserves_joint_statistics():
    vectors = np.random.default_rng(14).normal(size=(301, 7))
    moments = m.EventMoments(7)
    for chunk in (vectors[:128], vectors[128:256], vectors[256:]):
        moments.update_batch(chunk)
    np.testing.assert_allclose(moments.total, vectors.sum(0), atol=1e-13)
    np.testing.assert_allclose(moments.outer, vectors.T @ vectors, atol=1e-12)
    assert moments.events == 301


@pytest.mark.parametrize(
    "problem", ["reused_source", "wrong_cohort", "duplicate_design", "bad_bin", "empty_reference", "changed_edges"]
)
def test_invalid_export_inputs_fail_without_output(tmp_path, problem):
    manifest = export_case(tmp_path)
    if problem == "reused_source":
        manifest["cohorts"]["aa"] *= 2
    elif problem == "wrong_cohort":
        manifest["cohorts"]["other"] = manifest["cohorts"].pop("aa")
    elif problem == "duplicate_design":
        manifest["design_ids"] = [3, 3]
    elif problem == "bad_bin":
        manifest["features"][0]["xmax"] = 0.0
    elif problem == "empty_reference":
        manifest["reference"][0][0]["scale"] = 0.0
    else:
        original = np.load(manifest["cohorts"]["aa"][0])
        arrays = {key: original[key] for key in original.files}
        info = json.loads(str(arrays["metadata"]))
        info["layout"][0]["blocks"][0]["edges"] = [0.0, 0.5, 2.0]
        info["sources"] = [{"path": "distinct", "sha256": "synthetic_distinct_source"}]
        arrays["metadata"] = json.dumps(info)
        other = tmp_path / "other.npz"
        np.savez_compressed(other, **arrays)
        manifest["cohorts"]["aa"].append(str(other))
    output = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match=r"Repeated|Wrong|Invalid|Empty|layouts differ"):
        m.export(manifest, output)
    assert not output.exists()


def test_no_overwrite(tmp_path):
    path = tmp_path / "existing.npz"
    m._save(path, a=[1])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        m._save(path, a=[2])
    assert path.read_bytes() == before


def test_cli_uses_manifest_relative_inputs(tmp_path):
    manifest, _ = collect_case(tmp_path)
    for stream in manifest["streams"]:
        stream["file"] = Path(stream["file"]).name
    path = tmp_path / "collect.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "cli.npz"
    subprocess.run(
        [sys.executable, "-m", m.__name__, "collect", "--manifest", str(path), "--output", str(output)], check=True
    )
    assert np.load(output)["events"] == 4


def test_export_cli_uses_relative_cohort_files(tmp_path):
    manifest = export_case(tmp_path)
    manifest["cohorts"] = {name: [Path(path).name for path in paths] for name, paths in manifest["cohorts"].items()}
    path = tmp_path / "export.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "exported.npz"
    subprocess.run(
        [sys.executable, "-m", m.__name__, "export", "--manifest", str(path), "--output", str(output)], check=True
    )
    assert np.load(output)["covariance"].shape == (2, 2, 2, 2)


def test_writer_keeps_identity_on_empty_and_filled_events():
    source = Path(__file__).parents[1] / "jetscape_analysis/analysis/analyze_events_base_STAT.py"
    node = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "analyze_event_chunk"
    )
    namespace = {
        "np": np,
        "pd": pd,
        "time": SimpleNamespace(time=lambda: 0),
        "logger": SimpleNamespace(info=lambda *_: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)  # noqa: S102 - committed method, no caller code
    obj = SimpleNamespace(
        use_event_based_centrality=False,
        n_event_max=99,
        is_AA=False,
        output_event_list=[],
        cross_section_dict={},
        event_has_entries=bool,
    )
    obj.analyze_event = lambda event: setattr(
        obj, "observable_dict_event", {"x": [1.0]} if event["event_ID"] == 1 else {}
    )

    frame = pd.DataFrame(
        [
            {
                "event_ID": i,
                "vertex_x": i * 0.1,
                "vertex_y": -i * 0.1,
                "event_weight": 1.0,
                "pt_hat": 5.0,
                "cross_section": 2.0,
                "cross_section_error": 0.1,
            }
            for i in (0, 1)
        ]
    )
    namespace["analyze_event_chunk"](obj, frame)
    assert [row["event_ID"] for row in obj.output_event_list] == [0, 1]
    assert "event_weight" not in obj.output_event_list[0]
    assert obj.output_event_list[1]["x"] == [1.0]


def test_bayesian_loader_handoff_if_installed(tmp_path):
    noise = pytest.importorskip("bayesian.emulation.simulation_noise")
    manifest = export_case(tmp_path)
    path = tmp_path / "bank.npz"
    m.export(manifest, path)
    selected = noise.load_aligned_covariance(path, [17, 3], [("toy", 1.0, 2.0), ("toy", 0.0, 1.0)])
    with np.load(path) as z:
        np.testing.assert_array_equal(selected, z["covariance"][np.ix_([1, 0], [1, 0], [1, 0], [1, 0])])
