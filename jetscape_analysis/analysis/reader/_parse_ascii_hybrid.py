"""Parse hybrid ascii input files in chunks.

.. codeauthor:: Raymond Ehlers
"""

import functools
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt

import jetscape_analysis.analysis.reader._parse_ascii as parse_ascii_base
from jetscape_analysis.analysis.reader._parse_ascii import CrossSection, HeaderInfo

logger = logging.getLogger(__name__)


def parse_cross_section(line: str) -> CrossSection:
    """Parse cross section from within a Hybrid model event header.

    Args:
        line: Line containing the event header information.
    Returns:
        Cross section information
    """
    # Parse the string.
    values = line.split(" ")
    if len(values) == 12 and values[0] == "weight":
        ###################################
        # Cross section info specification:
        ###################################
        #
        # The cross section info is formatted as follows in the event header, with each entry separated by a ` ` character:
        # e.g. `weight 3.52953e-07 cross 6.34093 X 0.380728 Y 3.54109 cross_err 6.34093 pthat 410.271`
        #       0      1           2     3       4 5        6 7       8         9       10    11
        #
        # I _think_ the units are mb^-1
        info = CrossSection(
            value=float(values[3]),  # Cross section
            error=float(values[9]),  # Cross section error
        )
    else:
        _msg = f"Parsing of cross section failed: {values}"
        raise parse_ascii_base.FailedToParseHeader(_msg)

    return info


def _parse_header_hybrid_v1(
    line_one: str,
    line_two: str,
    n_particles: int,
) -> HeaderInfo:
    """Parse Hybrid model event header.

    The event header is two lines, so we need to take the full iterator, rather than

    The most common case is that it's a header, in which case we parse the line. If it's not a header,
    we also check if it's a cross section, which we note as having found at the end of the file via
    an exception.

    Args:
        f: The file iterator.
    Returns:
        The HeaderInfo information that was extracted.
    Raises:
        ReachedXSecAtEndOfFileException: If we find the cross section.
    """
    # Parse the string.
    values = line_one.split(" ")

    """
    #####################
    Header specification:
    #####################
    Example output:
    '''
    # event 0
    weight 3.52953e-07 cross 6.34093 X 0.380728 Y 3.54109 cross_err 6.34093 pthat 410.271
    -95.0595 -399.106 -131.206 0 1 -2
    95.0595 399.106 -287.312 0 2 -2
    -0.286097 3.79677 0.338279 0.13957 211 0
    -0.0662432 0.108651 0.492543 0.13957 -211 0
    ....
    end
    # event 1
    ...
    '''

    # Header format:
    - It starts with a "#', followed by the event number.
    - Next line contains all of our weights, etc
    """
    # Parse the first line: e.g. `# event 0`
    # Validation
    if values[0] != "#" or values[1] != "event":
        raise parse_ascii_base.FailedToParseHeader(line_one)
    # And extract relevant values from the first line
    event_number = int(values[-1])

    # Now onto the second line:
    # e.g. `weight 3.52953e-07 cross 6.34093 X 0.380728 Y 3.54109 cross_err 6.34093 pthat 410.271`
    #       0      1           2     3       4 5        6 7       8         9       10    11
    values = line_two.split(" ")

    info = HeaderInfo(
        event_number=event_number,  # Event number
        event_plane_angle=-999.0,  # EP angle
        n_particles=n_particles,  # Number of particles
        event_weight=float(values[1]),  # Event weight
        vertex_x=float(values[5]),  # x vertex
        vertex_y=float(values[7]),  # y vertex
        vertex_z=-999.0,  # z vertex
        centrality=-1.0,  # centrality
        pt_hat=float(values[11]),  # pt hat
    )

    return info  # noqa: RET504


# Register header parsing functions
_file_format_version_to_header_parser = {-1: _parse_header_hybrid_v1}


def event_by_event_generator(
    f: Iterator[str], parse_header_line: Callable[[str, str, int], HeaderInfo]
) -> Iterator[HeaderInfo | str]:
    """Event-by-event generator using the Hybrid model output file.

    It alternates back and forth from yielding headers to the particles and back.

    Args:
        f: The file iterator
        parse_header_line: Function to parse the
    Raises:
        ReachedEndOfFileException: We've hit the end of file.
    """
    # Our first line should be the header, which will be denoted by a "#".
    # Let the calling know if there are no events left due to exhausting the iterator.
    event_lines = []
    try:
        # First, let's look for the header:
        line = next(f)
        # Validate that we're starting with a header!
        if not line.startswith("#"):
            raise parse_ascii_base.LineIsNotHeader(line)
        # Great, we've found a header!
        event_lines.append(line)

        # Now let's work through the particles
        # NOTE: We can't immediately do a yield_from since we don't know how many particles there are
        #       in this event. Instead, we have to discover it ourselves, and then yield from there.
        for line in f:
            # If we've hit "end", then we're done!
            # NOTE: rstrip() removes the "\n"
            if line.rstrip() == "end":
                # logger.debug("Do not store end line")
                break
            # We don't want to store "end", so we put it after the check.
            event_lines.append(line)

        # Since we store the number of particles in the header, we need to wait to parse the header
        # until we determine how many particles there are in the event.
        header = parse_header_line(
            # The header consists of two lines, so we need to pass both of them.
            *event_lines[:2],
            # -2 due to the two header lines
            # NOTE: This count includes the outgoing partons...
            n_particles=len(event_lines) - 2,
        )

        # logger.info(f"header: {header}")
        yield header
    except StopIteration:
        # logger.debug("Hit end of file exception!")
        raise parse_ascii_base.ReachedEndOfFileException() from None

    # From the header, we know how many particles we have in the event, so we can
    # immediately yield the next n_particles lines. And this will leave the generator
    # at the next event, or (if we've exhausted the iterator) at the end of file.
    # NOTE: We header is stored in the first two lines, so we skip those when yielding.
    yield from event_lines[2:]


def initialize_parsing_functions(file_format_version: int) -> parse_ascii_base.ModelParameters:
    """Initialize parsing functions for the hybrid model.

    Args:
        file_format_version: Version of the file format.

    Returns:
        Functions to use for parsing the model output.
    """
    # Parser for the cross section and error
    # Search strategy:
    # The most accurate cross section and error determination is in the header of the
    # last event from the generation, so we need to search backwards from the end of the file
    # to the start of the last event. We'll do this by:
    # 1. Identify the start of the last event by finding the line starting with "#".
    #    This can take multiple steps, so we increase our read chunk size.
    # 2. Once we've scanned back far enough to find something, we'll ensure that we're
    #    looking at the right line by looking for the line containing "weight".
    extract_x_sec_func = functools.partial(
        parse_ascii_base.extract_x_sec_and_error,
        search_character_to_find_line_containing_cross_section="#",
        start_of_line_containing_cross_section="weight",
        parse_cross_section_line=parse_cross_section,
        read_chunk_size=1000,
    )

    # Event-by-event generator
    # All we need is the parser for the header lines
    e_by_e_generator = functools.partial(
        event_by_event_generator,
        parse_header_line=_file_format_version_to_header_parser[file_format_version],
    )

    return parse_ascii_base.ModelParameters(
        model_name="hybrid",
        column_names=["px", "py", "pz", "m", "particle_ID", "status"],
        extract_x_sec_and_error=extract_x_sec_func,
        event_by_event_generator=e_by_e_generator,
    )


def _parse_with_pandas(chunk_generator: Iterator[str]) -> npt.NDArray[Any]:
    """Parse the lines with `pandas.read_csv`

    `read_csv` uses a compiled c parser. As of 6 October 2020, it is tested to be the fastest option.

    Args:
        chunk_generator: Generator of chunks of the input file for parsing.
    Returns:
        Array of the particles.
    """
    # Delayed import so we only take the import time if necessary.
    # TODO: Try polars!
    import pandas as pd

    return pd.read_csv(  # type: ignore[call-overload,no-any-return]
        parse_ascii_base.FileLikeGenerator(chunk_generator),
        # NOTE: If the field is missing (such as eta and phi), they will exist, but they will be filled with NaN.
        #       We actively take advantage of this so we don't have to change the parsing for header v1 (which
        #       includes eta and phi) vs header v2 (which does not)
        # TODO: Need to update for hybrid
        # names=["particle_index", "particle_ID", "status", "E", "px", "py", "pz", "eta", "phi"],
        names=["px", "py", "pz", "m", "particle_ID", "status"],
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


def read(
    filename: Path | str, events_per_chunk: int, parser: str = "pandas", model: str = "jetscape"
) -> Iterator[ak.Array]:
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
    for i, chunk_generator in enumerate(
        read_events_in_chunks(filename=filename, events_per_chunk=events_per_chunk, model=model)
    ):
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
            "event_plane_angle": np.array([header.event_plane_angle for header in chunk_generator.headers], np.float32),
            "event_ID": np.array([header.event_number for header in chunk_generator.headers], np.uint32),
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
            header_level_info["pt_hat"] = np.array([header.pt_hat for header in chunk_generator.headers], np.float32)
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
        # TODO: Update to be compatible with hybrid and jetscape together!
        model_indices_and_types = {
            "jetscape": {
                "particle_ID": (1, np.int32),
                "status": (2, np.int8),
                "E": (3, np.float32),
                "px": (4, np.float32),
                "py": (5, np.float32),
                "pz": (6, np.float32),
            },
            "hybrid": {
                "particle_ID": (4, np.int32),
                "status": (5, np.int8),
                "px": (0, np.float32),
                "py": (1, np.float32),
                "pz": (2, np.float32),
                "m": (3, np.float32),
            },
        }
        yield ak.zip(
            {
                # Header level info
                **header_level_info,
                **{
                    k: ak.values_astype(array_with_events[:, :, v[0]], v[1])
                    for k, v in model_indices_and_types[model].items()
                },
                ## Particle level info
                # "particle_ID": ak.values_astype(array_with_events[:, :, 1], np.int32),
                ## We're only considering final state hadrons or partons, so status codes are limited to a few values.
                ## -1 are holes, while >= 0 are signal particles (includes both the jet signal and the recoils).
                ## So we can't differentiate the recoil from the signal.
                # "status": ak.values_astype(array_with_events[:, :, 2], np.int8),
                # "E": ak.values_astype(array_with_events[:, :, 3], np.float32),
                # "px": ak.values_astype(array_with_events[:, :, 4], np.float32),
                # "py": ak.values_astype(array_with_events[:, :, 5], np.float32),
                # "pz": ak.values_astype(array_with_events[:, :, 6], np.float32),
                ### We could skip eta and phi since we can always recalculate them. However, since we've already parsed
                ### them, we may as well pass them along.
                ##"eta": ak.values_astype(array_with_events[:, :, 7], np.float32),
                ##"phi": ak.values_astype(array_with_events[:, :, 8], np.float32),
            },
            depth_limit=1,
        )


if __name__ == "__main__":
    # For convenience
    logging.basicConfig(
        level=logging.DEBUG,
    )

    ## 0-5%
    # filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_05_Lres2_kappa_0p428/HYBRID_Hadrons_0000.out")
    ## 5-10%
    # filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_510_Lres2_kappa_0p428/HYBRID_Hadrons_0000.out")
    # Vacuum
    filename = Path("/Users/REhlers/software/dev/jetscape/hybrid-bayesian/HYBRID_Hadrons_Vac/HYBRID_Hadrons_0000.out")
    parse_to_parquet(
        base_output_filename=f"skim/{filename.parent.name}/{filename.stem}.parquet",
        store_only_necessary_columns=True,
        input_filename=filename,
        model="hybrid",
        # events_per_chunk=20,
        # max_chunks=3,
        events_per_chunk=100000,
        # max_chunks=1,
    )
