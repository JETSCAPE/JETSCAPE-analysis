"""Parse JETSCAPE ascii input files in chunks.

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
        model_parameters = _model_to_module[model].initialize_parsing_functions(file_format_version=file_format_version)
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


def _parse_with_pandas(chunk_generator: Iterator[str]) -> npt.NDArray[Any]:
    """Parse the lines with `pandas.read_csv`

    `read_csv` uses a compiled c parser. As of 6 October 2020, it is tested to be the fastest option.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
    Returns:
        Array of the particles.
    """
    # Delayed import so we only take the import time if necessary.
    import pandas as pd

    return pd.read_csv(  # type: ignore[no-any-return]
        FileLikeGenerator(chunk_generator),
        # NOTE: If the field is missing (such as eta and phi), they will exist, but they will be filled with NaN.
        #       We actively take advantage of this so we don't have to change the parsing for header v1 (which
        #       includes eta and phi) vs header v2 (which does not)
        names=["particle_index", "particle_ID", "status", "E", "px", "py", "pz", "eta", "phi"],
        header=None,
        comment="#",
        sep=r"\s+",
        # Converting to numpy makes the dtype conversion moot.
        # dtype={
        #     "particle_index": np.int32, "particle_ID": np.int32, "status": np.int8,
        #     "E": np.float32, "px": np.float32, "py": np.float32, "pz": np.float32,
        #     "eta": np.float32, "phi": np.float32
        # },
        # We can reduce the number of columns when reading.
        # However, it makes little difference, makes it less general, and we can always drop the columns later.
        # So we disable it for now.
        # usecols=["particle_ID", "status", "E", "px", "py", "eta", "phi"],
        #
        # NOTE: It's important that we convert to numpy before splitting. Otherwise, it will return columns names,
        #       which will break the header indexing and therefore the conversion to awkward.
    ).to_numpy()


def _parse_with_python(chunk_generator: Iterator[str]) -> npt.NDArray[Any]:
    """Parse the lines with python.

    We have this as an option because np.loadtxt is surprisingly slow.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
    Returns:
        Array of the particles.
    """
    particles = []
    for p in chunk_generator:
        if not p.startswith("#"):
            particles.append(np.array(p.rstrip("\n").split(), dtype=np.float64))
    return np.stack(particles)


def _parse_with_numpy(chunk_generator: Iterator[str]) -> npt.NDArray[Any]:
    """Parse the lines with numpy.

    Unfortunately, this option is surprisingly, presumably because it has so many options.
    Pure python appears to be about 2x faster. So we keep this as an option for the future,
    but it is not used by default.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
    Returns:
        Array of the particles.
    """
    return np.loadtxt(chunk_generator)


def read(filename: Path | str, events_per_chunk: int, parser: str = "pandas") -> Generator[ak.Array, int | None, None]:
    """Read a JETSCAPE FinalState{Hadrons,Partons} ASCII output file in chunks.

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

            # First, parse the lines. We need to make this call before attempt to convert into events because the necessary
            # info (namely, n particles per event) is only available and valid after we've parse the lines.
            res = parsing_function(iter(chunk_generator))

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
                "event_ID": np.array([header.event_number for header in chunk_generator.headers], np.uint16),
            }
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

            # Assemble all of the information in a single awkward array and pass it on.
            _result = yield ak.zip(
                {
                    # Header level info
                    **header_level_info,
                    # Particle level info
                    # As I've learned from experience, it's much more convenient to store the particles in separate columns.
                    # Trying to fit them in alongside the event level info makes life far more difficult.
                    #"particles": ak.zip(
                        **{
                            "particle_ID": ak.values_astype(array_with_events[:, :, 1], np.int32),
                            # We're only considering final state hadrons or partons, so status codes are limited to a few values.
                            # -1 are holes, while >= 0 are signal particles (includes both the jet signal and the recoils).
                            # So we can't differentiate the recoil from the signal.
                            "status": ak.values_astype(array_with_events[:, :, 2], np.int8),
                            "E": ak.values_astype(array_with_events[:, :, 3], np.float32),
                            "px": ak.values_astype(array_with_events[:, :, 4], np.float32),
                            "py": ak.values_astype(array_with_events[:, :, 5], np.float32),
                            "pz": ak.values_astype(array_with_events[:, :, 6], np.float32),
                            # We could skip eta and phi since we can always recalculate them. However, since we've already parsed
                            # them, we may as well pass them along.
                            "eta": ak.values_astype(array_with_events[:, :, 7], np.float32),
                            "phi": ak.values_astype(array_with_events[:, :, 8], np.float32),
                        },
                    #),
                },
                depth_limit=1,
            )

            # Update for next step
            chunk_generator = event_generator.send(_result)
            i += 1
    except StopIteration:
        pass


def full_events_to_only_necessary_columns_E_px_py_pz(arrays: ak.Array) -> ak.Array:
    columns_to_drop = ["eta", "phi"]
    columns_to_keep = [field for field in ak.fields(arrays) if field not in columns_to_drop]
    return ak.zip(
        {
            column: arrays[column] for column in columns_to_keep
        }, depth_limit=1
    )

## def full_events_to_only_necessary_columns_E_px_py_pz(arrays: ak.Array) -> ak.Array:
##     """Reduce the number of columns to store.
##
##     Note:
##         This only drops particle columns because those are the ones that have redundant info.
##         This is fairly specialized, but fine for our purposes here.
##     """
##     columns_to_drop = ["eta", "phi"]
##     return ak.zip(
##         {
##             **{k: v for k, v in zip(ak.fields(arrays), ak.unzip(arrays), strict=True) if k != "particles"},
##             "particles": ak.zip(
##                 {
##                     name: arrays["particles", name]
##                     for name in ak.fields(arrays["particles"])
##                     if name not in columns_to_drop
##                 }
##             ),
##         },
##         depth_limit=1,
##     )


def parse_to_parquet(
    base_output_filename: Path | str,
    store_only_necessary_columns: bool,
    input_filename: Path | str,
    events_per_chunk: int,
    parser: str = "pandas",
    max_chunks: int = -1,
    compression: str = "zstd",
    compression_level: int | None = None,
) -> None:
    """Parse the JETSCAPE ASCII and convert it to parquet, (potentially) storing only the minimum necessary columns.

    Args:
        base_output_filename: Basic output filename. Should include the entire path.
        store_only_necessary_columns: If True, store only the necessary columns, rather than all of them.
        input_filename: Filename of the input JETSCAPE ASCII file.
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
        # Reduce to the minimum required data.
        if store_only_necessary_columns:
            arrays = full_events_to_only_necessary_columns_E_px_py_pz(arrays)  # noqa: PLW2901
        else:
            # To match the steps taken when reducing the columns, we'll re-zip with the depth limited to 1.
            # As of April 2021, I'm not certainly this is truly required anymore, but it may be needed for
            # parquet writing to be successful (apparently parquet couldn't handle lists of structs sometime
            # in 2020. The status in April 2021 is unclear, but not worth digging into now).
            arrays = ak.zip(dict(zip(ak.fields(arrays), ak.unzip(arrays), strict=True)), depth_limit=1)  # noqa: PLW2901

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


if __name__ == "__main__":
    # read(filename="final_state_hadrons.dat", events_per_chunk=-1, base_output_filename="skim/jetscape.parquet")
    for pt_hat_range in ["7_9", "20_25", "50_55", "100_110", "250_260", "500_550", "900_1000"]:
        print(f"Processing pt hat range: {pt_hat_range}")  # noqa: T201
        directory_name = "OutputFile_Type5_qhatA10_B0_5020_PbPb_0-10_0.30_2.0_1"
        filename = f"JetscapeHadronListBin{pt_hat_range}"
        parse_to_parquet(
            base_output_filename=f"skim/{filename}.parquet",
            store_only_necessary_columns=True,
            input_filename=f"/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/{directory_name}/{filename}_test.out",
            events_per_chunk=20,
            # max_chunks=3,
        )
