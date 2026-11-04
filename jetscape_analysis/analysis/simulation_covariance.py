"""Opt-in event-Poisson moments and labelled covariance export for Bayesian training.

Observable reconstruction, bin choices, independence cohorts and normalization
maps belong to the caller. This module has no model-specific reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


class EventMoments:
    """One update is one joint weighted event, including normalization entries."""

    def __init__(self, width):
        self.total = np.zeros(width)
        self.outer = np.zeros((width, width))
        self.events = 0

    def update(self, vector):
        vector = np.asarray(vector, float)
        if vector.shape != self.total.shape or not np.isfinite(vector).all():
            msg = "Expected a finite joint event vector"
            raise ValueError(msg)
        self.update_batch(vector[None, :])

    def update_batch(self, vectors):
        vectors = np.asarray(vectors, float)
        if vectors.ndim != 2 or vectors.shape[1:] != self.total.shape or not np.isfinite(vectors).all():
            msg = "Expected finite event-by-statistic vectors"
            raise ValueError(msg)
        self.total += vectors.sum(axis=0)
        self.outer += vectors.T @ vectors
        self.events += len(vectors)


def histogram_counts(values, edges):
    """ROOT [low, high) bins, with an independent underflow+visible counter."""
    values = np.asarray([] if values is None else values, float)
    edges = np.asarray(edges, float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        msg = "Expected finite scalar values and increasing bin edges"
        raise ValueError(msg)
    visible = values[(values >= edges[0]) & (values < edges[-1])]
    return np.histogram(visible, edges)[0], int(np.count_nonzero(values < edges[-1]))


def _layout(streams):
    labels = []
    names = [s["name"] for s in streams]
    if not names or len(set(names)) != len(names):
        msg = "Require distinct paired stream names"
        raise ValueError(msg)
    for stream in streams:
        if not stream["blocks"]:
            msg = "A stream needs at least one histogram block"
            raise ValueError(msg)
        for block in stream["blocks"]:
            histogram_counts([], block["edges"])
            if not block["columns"] or not all(np.isfinite(v) for v in block["columns"].values()):
                msg = "Require explicit finite column coefficients"
                raise ValueError(msg)
            labels.extend(f"{stream['name']}:{block['name']}:bin{i}" for i in range(len(block["edges"]) - 1))
            norm = block.get("normalization", "events")
            if norm not in ("events", "all_entries", "underflow_and_visible"):
                msg = "Unknown normalization counter"
                raise ValueError(msg)
            if norm != "events":
                labels.append(f"{stream['name']}:{block['name']}:norm")
        labels.append(f"{stream['name']}:events")
    if len(labels) != len(set(labels)):
        msg = "Duplicate raw statistic labels"
        raise ValueError(msg)
    return labels


def _event_vector(row, blocks):
    weight = row["event_weight_all"]
    if weight is None or not np.isfinite(weight) or weight < 0:
        msg = "Invalid all-event weight"
        raise ValueError(msg)
    if row.get("event_weight") is not None and row["event_weight"] != weight:
        msg = "Accepted-event and all-event weights differ"
        raise ValueError(msg)
    vector = []
    for block in blocks:
        counts = np.zeros(len(block["edges"]) - 1)
        normalization = 0.0
        mode = block.get("normalization", "events")
        for column, coefficient in block["columns"].items():
            values = row[column]
            current, underflow_visible = histogram_counts(values, block["edges"])
            counts += coefficient * current
            normalization += coefficient * (
                underflow_visible if mode == "underflow_and_visible" else len([] if values is None else values)
            )
        vector.extend(counts)
        if mode != "events":
            vector.append(normalization)
    return weight * np.r_[vector, 1.0]


def _event_identity(rows, fields, seen):
    keys = [tuple(row[k] for k in fields) for row in rows]
    event_id = rows[0]["event_ID"]
    if (
        any(not np.isfinite(k).all() for k in keys)
        or any(k != keys[0] for k in keys[1:])
        or event_id < 0
        or event_id != int(event_id)
        or event_id in seen
    ):
        msg = "Nonfinite, duplicated or mismatched paired event identity"
        raise ValueError(msg)
    return event_id


def collect(manifest, output):
    """Collect complete aligned parquet streams into one independent cohort chunk.

    Streams are paired by explicit saved identities, not inferred from equal
    weights. Independent chunks/centralities/reference samples are collected in
    separate calls; export sums their covariance, not their event vectors.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 - collection-only dependency

    streams = manifest["streams"]
    labels = _layout(streams)
    fields = manifest.get("identity_columns", ["event_ID"])
    if not fields or "event_ID" not in fields or len(set(fields)) != len(fields):
        msg = "Identity columns must include event_ID exactly once"
        raise ValueError(msg)
    cohort = manifest["cohort"]
    if not isinstance(cohort, str) or not cohort or "/" in cohort:
        msg = "Require a nonempty cohort name without /"
        raise ValueError(msg)
    sources, readers = [], []
    expected = None
    for stream in streams:
        path = Path(stream["file"]).resolve()
        sources.append({"path": str(path), "sha256": _hash(path)})
        file = pq.ParquetFile(path)
        columns = set(fields) | {"event_weight_all"}
        columns.update(c for b in stream["blocks"] for c in b["columns"])
        if not columns <= set(file.schema_arrow.names):
            msg = "Missing histogram/identity columns; do not guess a legacy event join"
            raise ValueError(msg)
        columns |= {"event_weight"} & set(file.schema_arrow.names)
        if expected is not None and file.metadata.num_rows != expected:
            msg = "Paired streams have different event counts"
            raise ValueError(msg)
        expected = file.metadata.num_rows
        readers.append(
            r for batch in file.iter_batches(batch_size=128, columns=sorted(columns)) for r in batch.to_pylist()
        )
    moments, identities, pending = EventMoments(len(labels)), set(), []
    for rows in itertools.zip_longest(*readers):
        if any(row is None for row in rows):
            msg = "Paired streams ended at different positions"
            raise ValueError(msg)
        event_id = _event_identity(rows, fields, identities)
        identities.add(event_id)
        pending.append(np.concatenate([_event_vector(row, s["blocks"]) for row, s in zip(rows, streams, strict=True)]))
        if len(pending) == 128:
            moments.update_batch(pending)
            pending.clear()
    if pending:
        moments.update_batch(pending)
    if moments.events != expected or not moments.events:
        msg = "Incomplete or empty cohort"
        raise ValueError(msg)
    if any(_hash(source["path"]) != source["sha256"] for source in sources):
        msg = "An input changed during extraction"
        raise ValueError(msg)
    metadata = {
        "cohort": cohort,
        "sources": sources,
        "identity_columns": fields,
        "layout": [{k: v for k, v in s.items() if k != "file"} for s in streams],
        "method": "Uncentered weighted event-Poisson moments, including denominator fluctuations",
    }
    _save(
        output,
        total=moments.total,
        outer=moments.outer,
        labels=labels,
        events=moments.events,
        metadata=json.dumps(metadata, sort_keys=True),
    )


def _load_totals(cohorts):
    totals, labels, records, used = [], [], {}, set()
    for name, paths in cohorts.items():
        if not paths or "/" in name:
            msg = "Require nonempty independent cohort chunks"
            raise ValueError(msg)
        total = layout = metadata = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                info = json.loads(str(data["metadata"]))
                current = data["total"]
                current_layout = data["labels"].astype(str).tolist()
                if info["cohort"] != name or int(data["events"]) <= 0:
                    msg = "Wrong or empty cohort"
                    raise ValueError(msg)
                identities = {source["sha256"] for source in info["sources"]}
                if not identities or used & identities:
                    msg = "Repeated input sources cannot be independent chunks/cohorts"
                    raise ValueError(msg)
                used.update(identities)
                if metadata is not None and (
                    current_layout != layout
                    or info["layout"] != metadata["layout"]
                    or info["identity_columns"] != metadata["identity_columns"]
                ):
                    msg = "Chunk histogram or identity layouts differ"
                    raise ValueError(msg)
                if current.shape != (len(current_layout),) or not np.isfinite(current).all():
                    msg = "Invalid chunk totals"
                    raise ValueError(msg)
                total = current.copy() if total is None else total + current
                layout, metadata = current_layout, info
            records[str(Path(path).resolve())] = _hash(path)
        totals.append(total)
        labels.extend(f"{name}/{label}" for label in layout)
    if not totals or len(labels) != len(set(labels)):
        msg = "Require distinct raw statistic identities"
        raise ValueError(msg)
    return np.concatenate(totals), labels, records


def spectrum_map(total, labels, rows):
    """Explicit sums of ratios of linear raw statistics; return values and Jacobian."""
    from scipy.sparse import csr_matrix  # noqa: PLC0415 - optional export extra

    lookup = {label: i for i, label in enumerate(labels)}
    values, derivatives = [], []
    for terms in rows:
        if not terms:
            msg = "Each spectrum bin needs an explicit normalization term"
            raise ValueError(msg)
        value, derivative = 0.0, {}
        for term in terms:
            maps = []
            for key in ("numerator", "denominator"):
                mapping = {lookup[label]: float(coefficient) for label, coefficient in term[key].items()}
                if not mapping or not np.isfinite(list(mapping.values())).all():
                    msg = "Invalid linear ratio map"
                    raise ValueError(msg)
                maps.append(mapping)
            a, b = maps
            num = sum(coefficient * total[i] for i, coefficient in a.items())
            den = sum(coefficient * total[i] for i, coefficient in b.items())
            scale = float(term["scale"])
            if den <= 0 or not np.isfinite([num, den, scale]).all():
                msg = "Nonfinite ratio or nonpositive normalization"
                raise ValueError(msg)
            value += scale * num / den
            for i in a.keys() | b.keys():
                derivative[i] = derivative.get(i, 0.0) + scale * (a.get(i, 0.0) / den - num * b.get(i, 0.0) / den**2)
        values.append(value)
        derivatives.append(derivative)
    rr, cc, vv = [], [], []
    for row, derivative in enumerate(derivatives):
        for col, value in derivative.items():
            rr.append(row)
            cc.append(col)
            vv.append(value)
    return np.asarray(values), csr_matrix((vv, (rr, cc)), shape=(len(rows), len(total)))


def export(manifest, output):
    """Write the design,bin,design,bin NPZ contract consumed by JETSCAPE/Bayesian."""
    from scipy.sparse import vstack  # noqa: PLC0415 - optional export extra

    ids = np.asarray(manifest["design_ids"])
    features = manifest["features"]
    identities = [(f["name"], float(f["xmin"]), float(f["xmax"])) for f in features]
    if (
        ids.ndim != 1
        or not len(ids)
        or not np.issubdtype(ids.dtype, np.integer)
        or len(set(ids)) != len(ids)
        or np.any(ids < 0)
        or not identities
        or len(set(identities)) != len(identities)
        or any(not name or not np.isfinite([lo, hi]).all() or lo >= hi for name, lo, hi in identities)
    ):
        msg = "Invalid design IDs or observable/bin identities"
        raise ValueError(msg)
    spectra = manifest["spectra"]
    if len(spectra) != len(ids) or any(len(row) != len(features) for row in spectra):
        msg = "Spectrum shape does not match design and feature order"
        raise ValueError(msg)
    if len(manifest["reference"]) != len(features):
        msg = "Reference shape differs from feature order"
        raise ValueError(msg)
    total, labels, records = _load_totals(manifest["cohorts"])
    aa, ja = spectrum_map(total, labels, [bin_terms for design in spectra for bin_terms in design])
    pp, jp = spectrum_map(total, labels, manifest["reference"])
    if np.any(pp <= 0):
        msg = "Empty/nonpositive reference bin; do not invent an error"
        raise ValueError(msg)
    f = len(features)
    denominator = np.tile(pp, len(ids))
    means = aa / denominator
    gradient = ja.multiply(1 / denominator[:, None]) - vstack([jp] * len(ids)).multiply((aa / denominator**2)[:, None])
    gradient = gradient.tocsr()
    covariance = np.zeros((len(means), len(means)))
    offset = 0
    for paths in manifest["cohorts"].values():
        for path in paths:
            if _hash(path) != records[str(Path(path).resolve())]:
                msg = "Moment input changed during export"
                raise ValueError(msg)
            with np.load(path, allow_pickle=False) as data:
                outer = data["outer"]
                width = len(data["total"])
                if (
                    outer.shape != (width, width)
                    or not np.isfinite(outer).all()
                    or not np.allclose(outer, outer.T, rtol=1e-10, atol=0)
                ):
                    msg = "Invalid raw event outer moments"
                    raise ValueError(msg)
                local = gradient[:, offset : offset + width]
                covariance += np.asarray((local @ outer) @ local.T)
        offset += width
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(covariance).all()
        or np.max(np.abs(covariance - covariance.T)) > 1e-10 * scale
        or np.linalg.eigvalsh(covariance).min() < -1e-10 * scale
    ):
        msg = "Invalid/non-PSD propagated covariance; no repair was applied"
        raise ValueError(msg)
    _save(
        output,
        design_ids=ids,
        observable_names=[row[0] for row in identities],
        xmin=[row[1] for row in identities],
        xmax=[row[2] for row in identities],
        central_values=means.reshape(len(ids), f),
        covariance=covariance.reshape(len(ids), f, len(ids), f),
        provenance=json.dumps(
            {
                "inputs": records,
                "manifest": manifest,
                "method": "First-order ratio propagation of paired weighted event-Poisson moments; fixed map scales",
            },
            sort_keys=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "export"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for stream in manifest.get("streams", []):
        stream["file"] = str((args.manifest.parent / stream["file"]).resolve())
    for name, paths in manifest.get("cohorts", {}).items():
        manifest["cohorts"][name] = [str((args.manifest.parent / path).resolve()) for path in paths]
    {"collect": collect, "export": export}[args.stage](manifest, args.output)


if __name__ == "__main__":
    main()
