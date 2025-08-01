"""Parse ascii final state {hadrons,partons} files into (chunked) parquet files.

This module is designed to support output files from multiple models.
The model specific functionality is factored out into individual models,
which must implement the following API:

- extract_x_sec_and_error(f: typing.TextIO, read_chunk_size: int) -> CrossSection | None:
    Function to extract the cross section and cross section error from a file-like object.
- event_by_event_generator(f: Iterator[str], parse_header_line: Callable[[Any], HeaderInfo]) -> Iterator[HeaderInfo | str]:
    Generator to yield event-by-event information, switching back and forth both
    between headers and event particles.
- initialize_parsing_functions(file_format_version: int) -> ModelParameters:
    Initialize the model parameters and parsing functions needed to successfully
    parse an output file.

Once the module implements this API, it can be included in the following functions
to be included as a parsing option:

- determine_model_from_file(f: typing.TextIO) -> str:
    Determine the model based on the file contents. Customize to identify each
    model output.
- determine_format_version_from_file(f: typing.TextIO) -> int:
    Determine the file format version based on the file contents. Customize to
    identify each model output.
- Register the module in the _model_to_module dict

Based on this information, the model will automatically be determined and the output
will be parsed to a standardized and compressed parquet format for further analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt
import vector

import jetscape_analysis.analysis.reader._parse_ascii as parse_ascii_base
from jetscape_analysis.analysis.reader import _parse_ascii_hybrid, _parse_ascii_jetscape
from jetscape_analysis.analysis.reader._parse_ascii import ChunkGenerator

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PER_CHUNK_SIZE = int(1e5)


class UnrecognizedFileFormat(Exception):
    """Indicates that we cannot determine the file format."""


def determine_model_from_file(f: typing.TextIO) -> str:
    """Determine which model create the output file based on peeking at the beginning of the file.

    Args:
        f: The file iterator.

    Returns:
        Name of the model that was used to create the output file..
    """
    # Default to jetscape since it's most common.
    model_which_created_file = "jetscape"

    with parse_ascii_base.save_file_position(f):
        # For both jetscape and hybrid, the file should start with a comment.
        line_1 = next(f)
        if not line_1.startswith("#"):
            msg = f"Unable to determine file type due to unexpected file structure. Expected to start file with comment, but received {line_1}"
            raise UnrecognizedFileFormat(msg)
        # For the next line, JETSCAPE immediately goes into the particles, while the Hybrid has a second header line.
        line_2 = next(f)
        if line_2.startswith("weight"):
            model_which_created_file = "hybrid"

    return model_which_created_file


def determine_format_version_from_file(f: typing.TextIO) -> int:
    """Determine the file format version from the file.

    If the version cannot be identified, it defaults to -1.

    Args:
        f: File-like object.
    Returns:
        The version of the file format.
    """
    # Setup
    file_format_version = -1

    with parse_ascii_base.save_file_position(f):
        # Check for the file format version indicating how we should parse it.
        first_line = next(f)
        first_line_split = first_line.split("\t")
        if len(first_line_split) > 3 and first_line_split[1] == "JETSCAPE_FINAL_STATE":
            # 1: is to remove the "v" in the version
            file_format_version = int(first_line_split[2][1:])

        # If we haven't found the version, then there's nothing we can do

    return file_format_version


# Register model specific parsing functions.
_model_to_module = {
    "jetscape": _parse_ascii_jetscape,
    "hybrid": _parse_ascii_hybrid,
}


def read_events_in_chunks(
    filename: Path, events_per_chunk: int = DEFAULT_EVENTS_PER_CHUNK_SIZE
) -> Generator[ChunkGenerator, int | None, None]:
    """Read events in chunks from stored ASCII output files.

    This provides access to the lines of the file itself, but it is up to the user to parse each line.
    Consequently, many useful features are implemented on top of it. Users are encouraged to use those
    more full featured functions, such as `read(...)`.

    Args:
        filename: Path to the file.
        events_per_chunk: Number of events to store in each chunk. Default: 1e5.
    Returns:
        Chunks iterator. When this iterator is consumed, it will generate lines from the file until it
            hits the number of events mark. The header information is contained inside the object.
    """
    # Validation
    filename = Path(filename)

    with filename.open() as f:
        # First step, extract the model and the file format version
        model = determine_model_from_file(f)
        file_format_version = determine_format_version_from_file(f)
        logger.info(f"Found {model=}, {file_format_version=}")

        # And use that information to determine the parsing functions.
        # Validation
        if model not in _model_to_module:
            _msg = f"No parsing module found for {model=}"
            raise RuntimeError(_msg)
        model_parameters = _model_to_module[model].initialize_model_parameters(file_format_version=file_format_version)
        # And use those parsing functions to extract the final cross section and header.
        cross_section = model_parameters.extract_x_sec_and_error(f)

        # Now that we've complete the setup, we can move to actually parsing the events.
        # Define an iterator so we can increment it in different locations in the code.
        # `readlines()` is fine to use if it the entire file fits in memory.
        # read_lines = iter(f.readlines())
        # Use the iterator if the file doesn't fit in memory (fairly likely for these type of files)
        read_lines = iter(f)

        # Now, need to setup chunks.
        # NOTE: The headers and additional info are passed through the ChunkGenerator.
        requested_chunk_size: int | None = events_per_chunk
        while True:
            # This check is only needed for subsequent iterations - not the first
            if requested_chunk_size is None:
                requested_chunk_size = events_per_chunk

            # We keep an explicit reference to the chunk so we can set the end of file state
            # if we reached the end of the file.
            chunk = parse_ascii_base.ChunkGenerator(
                g=read_lines,
                events_per_chunk=requested_chunk_size,
                model_parameters=model_parameters,
                cross_section=cross_section,
            )
            requested_chunk_size = yield chunk
            if chunk.reached_end_of_file:
                break


class FileLikeGenerator:
    """Wrapper class to make a generator look like a file.

    Pandas requires passing a filename or a file-like object, but we handle the generator externally
    so we can find each chunk boundary, parse the headers, etc. Consequently, we need to make this
    generator appear as if it's a file.

    Based on https://stackoverflow.com/a/18916457/12907985

    Args:
        g: Generator to be wrapped.
    """

    def __init__(self, g: Iterator[str]):
        self.g = g

    def read(self, n: int = 0) -> Any:  # noqa: ARG002
        """Read method is required by pandas."""
        try:
            return next(self.g)
        except StopIteration:
            return ""

    def __iter__(self) -> Iterator[str]:
        """Iteration is required by pandas."""
        return self.g


# Defines the required column order, which standardizes how we write out data.
DEFAULT_EXPECTED_COLUMN_ORDER = [
    "particle_ID",
    "status",
    "E",
    "px",
    "py",
    "pz",
]
# All possible columns with their corresponding dtypes.
# NOTE: There could be overlapping columns here. It's not meant to provide
#       a consistent set of columns - it's just to provide the dtype mapping
#       for any possible columns.
COLUMN_TO_DTYPE = {
    "particle_ID": np.int32,
    "status": np.int8,
    "particle_index": np.uint16,
    "E": np.float32,
    "m": np.float32,
    "px": np.float32,
    "py": np.float32,
    "pz": np.float32,
    "eta": np.float32,
    "phi": np.float32,
}


def _parse_with_pandas(chunk_generator: Iterator[str], column_names: list[str]) -> npt.NDArray[Any]:
    """Parse the lines with `pandas.read_csv`

    `read_csv` uses a compiled c parser. As of 6 October 2020, it is tested to be the fastest option.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
        column_names: Name of the columns that are stored in the file.
    Returns:
        Array of the particles.
    """
    # Delayed import so we only take the import time if necessary.
    import pandas as pd  # noqa: PLC0415

    # If we have mass rather than E, replace it in the expected order.
    # We'll handle the conversion to write out the energy later
    expected_column_order = list(DEFAULT_EXPECTED_COLUMN_ORDER)
    if "E" not in column_names:
        expected_column_order[expected_column_order.index("E")] = "m"

    return pd.read_csv(  # type: ignore[no-any-return,call-overload]
        FileLikeGenerator(chunk_generator),
        # NOTE: If a field that is listed here is missing (e.g. eta or phi), they will be created and filled with NaN.
        #       We could take advantage of this so we don't have to change the parsing for jetscape header v1 (which
        #       includes eta and phi) vs header v2 (which does not)
        names=column_names,
        header=None,
        comment="#",
        sep=r"\s+",
        # NOTE: We do not set a dtype because it actually makes parsing slow (checked with timeit).
        #       Since we convert to numpy at the end (and then cast the dtype when we do the
        #       awkward array), there's no need to set the dtype now.
        # We could reduce the number of columns when reading.
        # However, it makes little difference, makes it less general, and we can always drop the columns later.
        # So we disable it for now.
        # usecols=["particle_ID", "status", "E", "px", "py", "eta", "phi"],
        #
        # NOTE: It's important that we convert to numpy before splitting. Otherwise, it will return columns names,
        #       which will break the header indexing and therefore the conversion to awkward.
    )[expected_column_order].to_numpy()


def _parse_with_polars(chunk_generator: Iterator[str], column_names: list[str]) -> npt.NDArray[Any]:
    """Parse the lines with polars.

    Note that this approach is quite hacky, but it's unclear how to parse without finding the
    event start indices, so we'll keep the current approach.

    NOTE:
        As of July 2025, RJE found this approach is about 30-40% faster than pandas for
        chunks of 5000-10000.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
        column_names: Name of the columns that are stored in the file.
    Returns:
        Array of the particles.
    """
    # Delayed import so we only take the import time if necessary.
    import polars as pl  # noqa: PLC0415

    # Polars need polars data types
    # NOTE: This isn't comprehensive, but covers our most common types
    _numpy_dtypes_to_polars_data_types = {
        np.int8: pl.Int8,
        np.uint8: pl.UInt8,
        np.int16: pl.Int16,
        np.uint16: pl.UInt16,
        np.int32: pl.Int32,
        np.uint32: pl.UInt32,
        np.int64: pl.Int64,
        np.uint64: pl.UInt64,
        np.float32: pl.Float32,
        np.float64: pl.Float64,
    }
    # Determine the schema
    schema = {k: _numpy_dtypes_to_polars_data_types[COLUMN_TO_DTYPE[k]] for k in column_names}

    # If we have mass rather than E, replace it in the expected order.
    # We'll handle the conversion to write out the energy later
    # NOTE: This is implicitly handled in the schema since the keys are
    #       set by the column names, which will already have e.g. the mass
    expected_column_order = list(DEFAULT_EXPECTED_COLUMN_ORDER)
    if "E" not in column_names:
        expected_column_order[expected_column_order.index("E")] = "m"

    # Begin the parsing process.
    # First, we need everything all of the lines in the chunk to be available in to get started.
    raw_lines = list(chunk_generator)
    # NOTE: By using `list(...)`, if we hit a StopIteration immediately (i.e. this means that we've
    #       reached the end of the file), we'll just get an empty list. The empty list will cause problems
    #       when trying to initialize the DataFrame, so we need to explicitly handle the case first.
    if not raw_lines:
        # Pass on that we've reached the end of the file.
        raise parse_ascii_base.ReachedEndOfFileException()

    # Now we have content, so we can parse it.
    # An explanation of the steps here:
    return (
        pl.DataFrame(
            # 1. Ingest the generator into a dataframe in the column "raw_lines"
            {"raw_lines": raw_lines}
        )
        .with_columns(
            # 2. Parse raw_lines by stripping whitespace (strip_chars) and splitting into exactly `len(column_names) - 1`
            #    splits. We need the -1 since e.g. 6 columns have 5 spaces in between them.
            pl.col("raw_lines").str.strip_chars().str.split_exact(" ", n=len(column_names) - 1).alias("split")
        )
        .unnest(
            # 3. Splits the parsed columns (in a struct) into separate columns for each field
            "split"
        )
        .rename(
            # 4. Renames the default column names (assigned by unnest) into their proper names
            {f"field_{i}": name for i, name in enumerate(list(schema))}
        )
        .cast(
            # 5. Casts them to their proper dtype
            schema  # type: ignore[arg-type]
            # 6. And selects only the columns that we would like to return, putting them in
            #    the expected order.
            # NOTE: It's important that we convert to numpy before splitting. Otherwise, it will return columns names,
            #       which will break the header indexing and therefore the conversion to awkward.
        )[expected_column_order]
        .to_numpy()
    )

    # The alternative approach below uses the pl.DataFrame constructor, but parses the strings in python.
    # It's substantially slower (e.g. factor of two slower), so better to stick with the above!!
    # return pl.DataFrame(
    #     # We need to parse the space separate values by hand. This is probably not particularly fast...
    #     # NOTE: This only handles spaces (i.e. not general whitespace like tabs)
    #     [v.rstrip().split(" ") for v in chunk_generator],
    #     schema=schema,
    #     orient="row",
    #     # NOTE: It's important that we convert to numpy before splitting. Otherwise, it will return columns names,
    #     #       which will break the header indexing and therefore the conversion to awkward.
    # )[expected_column_order].to_numpy()


def _parse_with_python(chunk_generator: Iterator[str], column_names: list[str]) -> npt.NDArray[Any]:  # noqa: ARG001
    """Parse the lines with python.

    We have this as an option because np.loadtxt is surprisingly slow.

    NOTE:
        This only works for jetscape generated files!

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
        column_names: Name of the columns that are stored in the file.
    Returns:
        Array of the particles.
    """
    particles = []
    for p in chunk_generator:
        if not p.startswith("#"):
            particles.append(np.array(p.rstrip("\n").split(), dtype=np.float64))
    return np.stack(particles)


def _parse_with_numpy(chunk_generator: Iterator[str], column_names: list[str]) -> npt.NDArray[Any]:  # noqa: ARG001
    """Parse the lines with numpy.

    Unfortunately, this option is surprisingly slow, presumably because it has so many options.
    Pure python appears to be about 2x faster. So we keep this as an option for the future,
    but it is not used by default.

    NOTE:
        This only works for jetscape generated files!

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
        column_names: Name of the columns that are stored in the file.
    Returns:
        Array of the particles.
    """
    return np.loadtxt(chunk_generator)


def read(filename: Path | str, events_per_chunk: int, parser: str = "pandas") -> Generator[ak.Array, int | None, None]:  # noqa: C901
    """Read a FinalState{Hadrons,Partons} ASCII output file in chunks.

    This is the primary user function. We read in chunks to keep the memory usage manageable.

    Note:
        We store the data in the smallest possible types that can still encompass their range.

    Args:
        filename: Filename of the ASCII file.
        events_per_chunk: Number of events to provide in each chunk.
        parser: Name of the parser to use. Default: `pandas`, which uses `pandas.read_csv`. It uses
            compiled c, and seems to be the fastest available option. Other options: ["python", "numpy"].
    Returns:
        Generator of an array of events_per_chunk events.
    """
    # Validation
    filename = Path(filename)

    # Setup
    parsing_function_map = {
        "pandas": _parse_with_pandas,
        "polars": _parse_with_polars,
        "python": _parse_with_python,
        "numpy": _parse_with_numpy,
    }
    parsing_function = parsing_function_map[parser]

    # Read the file, creating chunks of events.
    event_generator = read_events_in_chunks(filename=filename, events_per_chunk=events_per_chunk)
    try:
        i = 0
        chunk_generator = next(event_generator)
        while True:
            # Give a notification just in case the parsing is slow...
            logger.debug(f"New chunk {i}")

            # Optionally skip the very first line containing the file format version information
            if i == 0 and chunk_generator.model_parameters.has_file_format_line_at_beginning_of_file:
                line_to_skip = next(chunk_generator.g)
                logger.debug(f"Skipping very first line, as requested. Line: {line_to_skip}")

            if chunk_generator._is_chunk_ready():
                logger.info(f"{chunk_generator.reached_end_of_file=}")

            # First, parse the lines. We need to make this call before attempt to convert into events because the necessary
            # info (namely, n particles per event) is only available and valid after we've parse the lines.
            # NOTE: There are certain parsers where we need to re-raise that we've reached the end of the file
            #       (this is only necessary for polars, as July 2025). We'll address this in the next clause just below here,
            #       so to keep a single breakout point, we just catch it here and do nothing.
            try:
                res = parsing_function(
                    iter(chunk_generator), column_names=chunk_generator.model_parameters.column_names
                )
            except parse_ascii_base.ReachedEndOfFileException:
                # Just log and keep going. See the note above.
                logger.debug(
                    f"Reached end of file and had to raise explicit exception. {chunk_generator.reached_end_of_file=}, {len(chunk_generator.headers)=}"
                )

            # Before we do anything else, if our events_per_chunk is a even divisor of the total number of events
            # and we've hit the end of the file, we can return an empty generator after trying to parse the chunk.
            # In that case, we're done - just break.
            # NOTE: In the case where we've reached the end of file, but we've parsed some events, we want to continue
            #       on so we don't lose any events.
            if chunk_generator.reached_end_of_file and len(chunk_generator.headers) == 0:
                break

            # Now, convert into the awkward array structure.
            # logger.info(f"n_particles_per_event len: {len(chunk_generator.n_particles_per_event())}, array: {chunk_generator.n_particles_per_event()}")
            array_with_events = ak.unflatten(ak.Array(res), chunk_generator.n_particles_per_event())

            # Cross checks.
            # Length check that we have as many events as expected based on the number of headers.
            # logger.debug(f"ak.num: {ak.num(array_with_events, axis = 0)}, len headers: {len(chunk_generator.headers)}")
            assert ak.num(array_with_events, axis=0) == len(chunk_generator.headers)
            # Check that n particles agree
            n_particles_from_header = np.array([header.n_particles for header in chunk_generator.headers])
            # logger.info(f"n_particles from headers: {n_particles_from_header}")
            # logger.info(f"n_particles from array: {ak.num(array_with_events, axis = 1)}")
            assert (np.asarray(ak.num(array_with_events, axis=1)) == n_particles_from_header).all()
            # State of the chunk
            # logger.debug(f"Reached end of file: {chunk_generator.reached_end_of_file}")
            # logger.debug(f"Incomplete chunk: {chunk_generator.incomplete_chunk}")
            # Let the use know so they're not surprised.
            if chunk_generator.incomplete_chunk:
                logger.warning(
                    f"Requested {chunk_generator.events_per_chunk} events, but only {chunk_generator.events_contained_in_chunk} are available because we hit the end of the file."
                )

            # Header info
            header_level_info = {
                "event_plane_angle": np.array(
                    [header.event_plane_angle for header in chunk_generator.headers], np.float32
                ),
                # NOTE: Since the event ID is continuous, a long running simulation could overflow uint16, so we take uint32
                "event_ID": np.array([header.event_number for header in chunk_generator.headers], np.uint32),
            }
            # And then handle optional values - no point storing the values if they're not meaningful.
            if chunk_generator.headers[0].event_weight > -1:
                header_level_info["event_weight"] = np.array(
                    [header.event_weight for header in chunk_generator.headers], np.float32
                )
            if chunk_generator.headers[0].centrality > -1:
                header_level_info["centrality"] = np.array(
                    [header.centrality for header in chunk_generator.headers], np.float32
                )
            if chunk_generator.headers[0].pt_hat > -1:
                header_level_info["pt_hat"] = np.array(
                    [header.pt_hat for header in chunk_generator.headers], np.float32
                )
            if chunk_generator.headers[0].vertex_x > -999:
                header_level_info["vertex_x"] = np.array(
                    [header.vertex_x for header in chunk_generator.headers], np.float32
                )
            if chunk_generator.headers[0].vertex_y > -999:
                header_level_info["vertex_y"] = np.array(
                    [header.vertex_y for header in chunk_generator.headers], np.float32
                )
            if chunk_generator.headers[0].vertex_z > -999:
                header_level_info["vertex_z"] = np.array(
                    [header.vertex_z for header in chunk_generator.headers], np.float32
                )

            # Cross section info
            if chunk_generator.cross_section:
                # Even though this is a dataset level quantity, we need to match the structure in order to zip them together for storage.
                # Since we're repeating the value, hopefully this will be compressed effectively.
                header_level_info["cross_section"] = np.full_like(
                    header_level_info["event_plane_angle"], chunk_generator.cross_section.value
                )
                header_level_info["cross_section_error"] = np.full_like(
                    header_level_info["event_plane_angle"], chunk_generator.cross_section.error
                )

            # Handle the particle level fields
            # Namely, this enables the conversion of particle level fields if necessary (e.g. m -> E)
            energy_like_label = "E"
            if "m" in chunk_generator.model_parameters.column_names:
                energy_like_label = "m"
            # Order of the columns is determined at the time of parsing
            particles = vector.Array(
                ak.zip(
                    {
                        "particle_ID": ak.values_astype(array_with_events[:, :, 0], COLUMN_TO_DTYPE["particle_ID"]),
                        # We're only considering final state hadrons or partons, so status codes are limited to a few values.
                        # -1 are holes, while >= 0 are signal particles (includes both the jet signal and the recoils).
                        # So we can't differentiate the recoil from the signal.
                        "status": ak.values_astype(array_with_events[:, :, 1], COLUMN_TO_DTYPE["status"]),
                        energy_like_label: ak.values_astype(
                            array_with_events[:, :, 2], COLUMN_TO_DTYPE[energy_like_label]
                        ),
                        "px": ak.values_astype(array_with_events[:, :, 3], COLUMN_TO_DTYPE["px"]),
                        "py": ak.values_astype(array_with_events[:, :, 4], COLUMN_TO_DTYPE["py"]),
                        "pz": ak.values_astype(array_with_events[:, :, 5], COLUMN_TO_DTYPE["pz"]),
                    },
                )
            )
            # Regardless of what kinematics were provided, we want to write E, px, py, pz.
            # These are contained in the DEFAULT_EXPECTED_COLUMN_ORDER, so we use that
            # here, event if we don't care overly much about the order once we've created
            # the awkward array (i.e. the ordering is not meaningful there)
            particle_fields_to_write = list(DEFAULT_EXPECTED_COLUMN_ORDER)

            # Assemble all of the information in a single awkward array and pass it on.
            _result = yield ak.zip(
                {
                    # Header level info
                    **header_level_info,
                    # Particle level info
                    # NOTE: We have to decompose this into a dict since we need to rezip
                    #       at a lower depth limit to match up with the header information.
                    #       (pandas doesn't properly read a fully zipped particles. If we're working
                    #       with awkward later, we can always rezip it at full depth then).
                    # NOTE: We need to use getattr since vector may not instantiate the field otherwise.
                    **{k: getattr(particles, k) for k in particle_fields_to_write},
                },
                depth_limit=1,
            )

            # Update for next step
            chunk_generator = event_generator.send(_result)
            i += 1
    except StopIteration:
        pass

    logger.info("Done with reading...")


def parse_to_parquet(
    base_output_filename: Path | str,
    input_filename: Path | str,
    events_per_chunk: int,
    parser: str = "polars",
    max_chunks: int = -1,
    compression: str = "zstd",
    compression_level: int | None = None,
) -> None:
    """Parse the ASCII final state {hadrons,partons} output file and convert it to parquet.

    Args:
        base_output_filename: Basic output filename. Should include the entire path.
        input_filename: Filename of the input ASCII final state {hadrons,partons} file.
        events_per_chunk: Number of events to be read per chunk.
        parser: Name of the parser. Default: "pandas".
        max_chunks: Maximum number of chunks to read. Default: -1.
        compression: Compression algorithm for parquet. Default: "zstd". Options include: ["snappy", "gzip", "ztsd"].
            "gzip" is slightly better for storage, but slower. See the compression tests and parquet docs for more.
        compression_level: Compression level for parquet. Default: `None`, which lets parquet choose the best value.
    Returns:
        None. The parsed events are stored in parquet files.
    """
    # Validation
    base_output_filename = Path(base_output_filename)
    # Setup the base output directory
    base_output_filename.parent.mkdir(parents=True, exist_ok=True)

    for i, arrays in enumerate(read(filename=input_filename, events_per_chunk=events_per_chunk, parser=parser)):
        # If converting in chunks, add an index to the output file so the chunks don't overwrite each other.
        if events_per_chunk > 0:
            suffix = base_output_filename.suffix
            output_filename = (base_output_filename.parent / f"{base_output_filename.stem}_{i:02}").with_suffix(suffix)
        else:
            output_filename = base_output_filename

        # Parquet with zlib seems to do about the same as ascii tar.gz when we drop unneeded columns.
        # And it should load much faster!
        ak.to_parquet(
            arrays,
            destination=str(output_filename),
            compression=compression,
            compression_level=compression_level,
            # Optimize the compression via improved encodings for floats and strings.
            # Conveniently, awkward 2.x will now select the right columns for each if simply set to `True`
            # Optimize for columns with anything other than floats
            parquet_dictionary_encoding=True,
            # Optimize for columns with floats
            parquet_byte_stream_split=True,
        )

        # Break now so we don't have to read the next chunk.
        if (i + 1) == max_chunks:
            break

    logger.info("Done with parsing.")


if __name__ == "__main__":
    # Setup
    from jetscape_analysis.base import helpers

    helpers.setup_logging(level=logging.INFO)

    # read(filename="final_state_hadrons.dat", events_per_chunk=-1, base_output_filename="skim/jetscape.parquet")
    # for pt_hat_range in ["7_9", "20_25", "50_55", "100_110", "250_260", "500_550", "900_1000"]:
    #     logger.info(f"Processing pt hat range: {pt_hat_range}")
    #     directory_name = "OutputFile_Type5_qhatA10_B0_5020_PbPb_0-10_0.30_2.0_1"
    #     filename = f"JetscapeHadronListBin{pt_hat_range}"
    #     parse_to_parquet(
    #         base_output_filename=f"skim/{filename}.parquet",
    #         input_filename=f"/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/{directory_name}/{filename}_test.out",
    #         events_per_chunk=20,
    #         # max_chunks=3,
    #     )

    ## 0-5%
    # filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_05_Lres2_kappa_0p428/HYBRID_Hadrons_0000.out")
    ## 5-10%
    # filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_510_Lres2_kappa_0p428/HYBRID_Hadrons_0000.out")
    # Vacuum
    filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_Vac/HYBRID_Hadrons_0000.out")
    parse_to_parquet(
        base_output_filename=f"skim/{filename.parent.name}/{filename.stem}.parquet",
        input_filename=filename,
        events_per_chunk=20,
        max_chunks=3,
        parser="polars",
        # events_per_chunk=100000,
        # max_chunks=1,
    )
