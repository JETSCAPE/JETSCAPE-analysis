"""Hybrid model specific parsing functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

import functools
import logging
from collections.abc import Iterator
from typing import Protocol

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


class ParseHeaderLine(Protocol):
    """Interface for parsing a header line from a hybrid model output.

    Args:
        line_one: The first line of the header.
        line_two: The second line of the header.
        n_particles: Number of particles in the event.
    Returns:
        The HeaderInfo information that was extracted.
    """
    def __call__(self, line_one: str, line_two: str, *, n_particles: int) -> HeaderInfo: ...


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
        line_one: The first line of the header.
        line_two: The second line of the header.
        n_particles: Number of particles in the event.
    Returns:
        The HeaderInfo information that was extracted.
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
# Associated column names to file format version.
# This is needed since polars requires knowledge about all of the columns
# in the file, and that can potentially vary by version.
# NOTE: The order here must match the order in the ascii file! We use this to label the columns in file.
_file_format_version_to_column_names = {
    -1: ["px", "py", "pz", "m", "particle_ID", "status"],
}


def event_by_event_generator(
    f: Iterator[str], parse_header_line: ParseHeaderLine
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


def initialize_model_parameters(file_format_version: int) -> parse_ascii_base.ModelParameters:
    """Initialize model parameters and parsing functions for the Hybrid model.

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
        column_names=_file_format_version_to_column_names[file_format_version],
        extract_x_sec_and_error=extract_x_sec_func,
        event_by_event_generator=e_by_e_generator,
        has_file_format_line_at_beginning_of_file=False,
    )
