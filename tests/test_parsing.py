""" Tests for v2 parser.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Any

import awkward as ak
import pytest
from jetscape_analysis.analysis.reader import parse_ascii

_here = Path(__file__).parent

# Remap column names that have been renamed.
_rename_columns = {
    "hydro_event_id": "event_ID",
}

@pytest.mark.parametrize(
    "header_version",
    [1, 2],
    ids=["Header v1", "Header v2"]
)
@pytest.mark.parametrize(
    "events_per_chunk",
    [
        5, 16, 50, 5000,
    ], ids=["Multiple, divisible: 5", "Multiple, indivisible: 16", "Equal: 50", "Larger: 5000"]
)
def test_parsing(header_version: int, events_per_chunk: int) -> None:
    input_filename = _here / "parsing" / f"final_state_hadrons_header_v{header_version}.dat"

    for i, arrays in enumerate(parse_ascii.read(filename=input_filename, events_per_chunk=events_per_chunk, parser="pandas")):
        # Get the reference array
        # Create the reference arrays by checking out the parser v1 (e477e0277fa560f9aba82310c02da8177e61c9e4), setting
        # the chunk size in skim_ascii, and then calling:
        # $ python jetscape_analysis/analysis/reader/skim_ascii.py -i tests/parsing/final_state_hadrons_header_v1.dat -o tests/parsing/events_per_chunk_50/parser_v1_header_v1/test.parquet
        # NOTE: The final state hadron files won't exist when you check out that branch, so
        #       it's best to copy them for your existing branch.
        reference_arrays = ak.from_parquet(
            Path(f"{_here}/parsing/events_per_chunk_{events_per_chunk}/parser_v1_header_v1/test_{i:02}.parquet")
        )
        # There are more fields in v2 than in the reference arrays (v1), so only take those
        # that are present in reference for comparison.
        # NOTE: We have to compare the fields one-by-one because the shapes of the fields
        #       are different, and apparently don't broadcast nicely with `__eq__`
        for field in ak.fields(reference_arrays):
            new_field = _rename_columns.get(field, field)
            assert ak.all(reference_arrays[field] == arrays[new_field])

        # Check for cross section if header v2
        if header_version == 2:
            assert "cross_section" in ak.fields(arrays)
            assert "cross_section_error" in ak.fields(arrays)


@pytest.mark.parametrize(
    "header_version",
    [1, 2],
    ids=["Header v1", "Header v2"]
)
@pytest.mark.parametrize(
    "events_per_chunk",
    [
        5, 16, 50, 5000,
    ], ids=["Multiple, divisible: 5", "Multiple, indivisible: 16", "Equal: 50", "Larger: 5000"]
)
def test_parsing_with_parquet(header_version: int, events_per_chunk: int, tmp_path: Path) -> None:
    """Parse to parquet, read back, and compare."""
    input_filename = _here / "parsing" / f"final_state_hadrons_header_v{header_version}.dat"

    # Convert to chunks in a temp directory.
    base_output_filename = tmp_path / "test.parquet"
    parse_ascii.parse_to_parquet(base_output_filename=base_output_filename,
                                 input_filename=input_filename,
                                 events_per_chunk=events_per_chunk)

    output_filenames = tmp_path.glob("*.parquet")

    for i, output_filename in enumerate(sorted(output_filenames)):
        arrays = ak.from_parquet(output_filename)

        # Create the reference arrays by checking out the parser v1 (e477e0277fa560f9aba82310c02da8177e61c9e4), setting
        # the chunk size in skim_ascii, and then calling:
        # $ python jetscape_analysis/analysis/reader/skim_ascii.py -i tests/parsing/final_state_hadrons_header_v1.dat -o tests/parsing/events_per_chunk_50/parser_v1_header_v1/test.parquet
        # NOTE: The final state hadron files won't exist when you check out that branch, so
        #       it's best to copy them for your existing branch.
        reference_arrays = ak.from_parquet(
            Path(f"{_here}/parsing/events_per_chunk_{events_per_chunk}/parser_v1_header_v1/test_{i:02}.parquet")
        )
        # There are more fields in v2 than in the reference arrays (v1), so only take those
        # that are present in reference for comparison.
        # NOTE: We have to compare the fields one-by-one because the shapes of the fields
        #       are different, and apparently don't broadcast nicely with `__eq__`
        for field in ak.fields(reference_arrays):
            new_field = _rename_columns.get(field, field)
            assert ak.all(reference_arrays[field] == arrays[new_field])

        # Check for cross section if header v2
        if header_version == 2:
            assert "cross_section" in ak.fields(arrays)
            assert "cross_section_error" in ak.fields(arrays)


@pytest.mark.parametrize(
    "model_input",
    [("JETSCAPE", Path("final_state_hadrons_header_v2_with_file_version.dat")), ("Hybrid", Path("HYBRID_Med_0000.out"))],
    ids=["JETSCAPE v2 header", "Hybrid model"],
)
@pytest.mark.parametrize("parser", ["polars", "pandas"])
@pytest.mark.parametrize("events_per_chunk", [10, 30, 50])
def test_parsing_methods_and_models(model_input: tuple[str, Path], parser: str, events_per_chunk: int, tmp_path: Path) -> None:
    """Test parsing methods and models.

    We want to confirm that the parsing works for a variety of inputs:
    - Varied model outputs:
        - JETSCAPE
        - Hybrid model
    - Varied parsers:
        - pandas
        - polars
    - Varied events_per_chunk:
        - Small number (multiple of total number)
        - Small number (not a multiple of total number)
        - Matches total number of events (50)
    """
    # Setup
    model_name, input_filename = model_input
    filename = _here / "parsing" / input_filename

    # Test conversion.
    parse_ascii.parse_to_parquet(
        base_output_filename=tmp_path / f"{filename.stem}_{parser}.parquet",
        input_filename=filename,
        events_per_chunk=events_per_chunk,
        parser=parser,
    )

    # And check the reference.
    output_filenames = tmp_path.glob("*.parquet")

    for i, output_filename in enumerate(sorted(output_filenames)):
        arrays = ak.from_parquet(output_filename)

        # Create the reference arrays by running the below for each events_per_chunk value:
        # Jetscape:
        # $ python3 -m jetscape_analysis.analysis.reader.skim_ascii -i tests/parsing/final_state_hadrons_header_v2_with_file_version.dat -o tests/parsing/methods_and_models/jetscape/events_per_chunk_10/reference.parquet -n 10
        # Hybrid:
        # $ python3 -m jetscape_analysis.analysis.reader.skim_ascii -i tests/parsing/HYBRID_Med_0000.out -o tests/parsing/methods_and_models/hybrid/events_per_chunk_10/reference.parquet -n 10
        # NOTE: These were last (re)created in July 2025 by RJE
        ref_path = _here / f"parsing/methods_and_models/{model_name.lower()}/events_per_chunk_{events_per_chunk}/reference_{i:02}.parquet"
        reference_arrays = ak.from_parquet(
            ref_path
        )
        # NOTE: We have to compare the fields one-by-one because the shapes of the fields
        #       are different, and apparently don't broadcast nicely with `__eq__`
        for field in ak.fields(reference_arrays):
            assert ak.all(reference_arrays[field] == arrays[field])

        # Check for cross section
        assert "cross_section" in ak.fields(arrays)
        assert "cross_section_error" in ak.fields(arrays)


@pytest.mark.parametrize(
    "model_input",
    [("JETSCAPE", Path("final_state_hadrons_header_v2_with_file_version.dat")), ("Hybrid", Path("HYBRID_Med_0000.out"))],
    ids=["JETSCAPE v2 header", "Hybrid model"],
)
@pytest.mark.parametrize("parser", ["polars", "pandas"])
def test_parsing_benchmark(model_input: tuple[str, Path], parser: str, benchmark: Any, tmp_path: Path) -> None:
    """Benchmark parsing methods.

    We only test a subset of methods to determine which are most efficient.
    Since the benchmark approaching will repeated run (e.g. like timeit),
    there's no need to waste additional computation on e.g. varying the chunk size.
    """
    # Setup
    model_name, input_filename = model_input
    filename = _here / "parsing" / input_filename

    # Setup the benchmark
    # Group benchmark results by model
    # (it doesn't make sense to compare across model)
    benchmark.group = model_name

    benchmark(
        parse_ascii.parse_to_parquet,
        base_output_filename=tmp_path / f"{filename.stem}_{parser}.parquet",
        input_filename=filename,
        events_per_chunk=50,
        max_chunks=1,
        parser=parser,
    )

