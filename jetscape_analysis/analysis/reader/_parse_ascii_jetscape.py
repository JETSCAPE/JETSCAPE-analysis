"""JETSCAPE specific parsing functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import functools
import itertools
import logging
from collections.abc import Callable, Iterator

import jetscape_analysis.analysis.reader._parse_ascii as parse_ascii_base
from jetscape_analysis.analysis.reader._parse_ascii import CrossSection, HeaderInfo

logger = logging.getLogger(__name__)


def parse_cross_section(line: str) -> CrossSection:
    """Parse cross section from a line containing the information.

    Args:
        line: Line containing the cross section information.
    Returns:
        Cross section information.
    """
    # Parse the string.
    values = line.split("\t")
    if len(values) == 5 and values[1] == "sigmaGen":
        ###################################
        # Cross section info specification:
        ###################################
        # The cross section info is formatted as follows, with each entry separated by a `\t` character:
        # # sigmaGen 182.423 sigmaErr 11.234
        # 0 1        2       3        4
        #
        # I _think_ the units are mb^-1
        info = CrossSection(
            value=float(values[2]),  # Cross section
            error=float(values[4]),  # Cross section error
        )
    else:
        _msg = f"Parsing of cross section failed: {values}"
        raise parse_ascii_base.FailedToParseHeader(_msg)

    return info


def _parse_optional_header_values(optional_values: list[str]) -> tuple[float, float]:
    """Parse optional JETSCAPE header values.

    As of April 2025, the centrality and pt_hat are optionally included in the output.
    The centrality will always be before the pt_hat.

    Args:
        optional_values: Optional values parsed from splitting the lines. It is expected
            to contain both the name of the arguments *AND* the values themselves. So if
            the file contains one optional values, it will translate to `len(optional_values) == 2`.

    Returns:
        The optional values: (centrality, pt_hat). If they weren't provided in the values,
            they will be their default values.
    """
    pt_hat = -1.0
    centrality = -1.0

    n_optional_values = len(optional_values)

    if n_optional_values == 4:
        # Both centrality and pt_hat are present
        if optional_values[-4] == "centrality":
            centrality = float(optional_values[-3])
        if optional_values[-2] == "pt_hat":
            pt_hat = float(optional_values[-1])
    elif n_optional_values == 2:
        # Only one of centrality or pt_hat is present
        if optional_values[-2] == "centrality":
            centrality = float(optional_values[-1])
        elif optional_values[-2] == "pt_hat":
            pt_hat = float(optional_values[-1])
    # If there are no optional values, then there's nothing to be done!

    return centrality, pt_hat


def _parse_header_line_format_unspecified(line: str) -> HeaderInfo:
    """Parse line that is expected to be a header.

    The most common case is that it's a header, in which case we parse the line. If it's not a header,
    we also check if it's a cross section, which we note as having found at the end of the file via
    an exception.

    Args:
        line: Line to be parsed.
    Returns:
        The HeaderInfo information that was extracted.
    Raises:
        ReachedXSecAtEndOfFileException: If we find the cross section.
    """
    # Parse the string.
    values = line.split("\t")
    # Compare by length first so we can short circuit immediately if it doesn't match, which should
    # save some string comparisons.
    info: HeaderInfo | CrossSection
    if (len(values) == 19 and values[1] == "Event") or (len(values) == 17 and values[1] == "Event"):
        ##########################
        # Header v2 specification:
        ##########################
        # As of 20 April 2021, the formatting of the header has been improved.
        # This function was developed to parse it.
        # The header is defined as follows, with each entry separated by a `\t` character:
        # `# Event 1 weight 0.129547 EPangle 0.0116446 N_hadrons 236 | N  pid status E Px Py Pz Eta Phi`
        #  0 1     2 3      4        5       6         7         8   9 10 11  12    13 14 15 16 17  18
        #
        # NOTE: Everything after the "|" is just documentation for the particle entries stored below.
        #
        info = HeaderInfo(
            event_number=int(values[2]),  # Event number
            event_plane_angle=float(values[6]),  # EP angle
            n_particles=int(values[8]),  # Number of particles
            event_weight=float(values[4]),  # Event weight
        )
    elif len(values) == 9 and "Event" in values[2]:
        ##########################
        # Header v1 specification:
        ##########################
        # The header v1 specification is as follows, with ">" followed by same spaces indicating a "\t" character:
        # #>  0.0116446>  Event1ID>   236>pstat-EPx>  Py> Pz> Eta>Phi
        # 0   1           2           3   4           5   6   7   8
        #
        # NOTE: There are some difficulties in parsing this header due to inconsistent spacing.
        #
        # The values that we want to extract are:
        # index: Meaning
        #   1: Event plane angle. float, potentially in scientific notation.
        #   2: Event number, of the from "EventNNNNID". Can be parsed as val[5:-2] to generically extract `NNNN`. int.
        #   3: Number of particles. int.
        info = HeaderInfo(
            event_number=int(values[2][5:-2]),  # Event number
            event_plane_angle=float(values[1]),  # EP angle
            n_particles=int(values[3]),  # Number of particles
        )
    elif len(values) == 5 and values[1] == "sigmaGen":
        # If we've hit the cross section, and we're not doing the initial extraction of the cross
        # section, this means that we're at the last line of the file, and should notify as such.
        # NOTE: By raising with the cross section, we make it possible to retrieve it, even though
        #       we've raised an exception here.
        raise parse_ascii_base.ReachedXSecAtEndOfFileException(parse_cross_section(line))
    else:
        _msg = f"Parsing of comment line failed: {values}"
        raise ValueError(_msg)

    return info


def _parse_header_line_format_v2(line: str) -> HeaderInfo:
    """Parse line that is expected to be a header according to the v2 file format.

    The most common case is that it's a header, in which case we parse the line. If it's not a header,
    we also check if it's a cross section, which we note as having found at the end of the file via
    an exception.

    Args:
        line: Line to be parsed.
    Returns:
        The HeaderInfo information that was extracted.
    Raises:
        ReachedXSecAtEndOfFileException: If we find the cross section.
    """
    # Parse the string.
    values = line.split("\t")
    # Compare by length first so we can short circuit immediately if it doesn't match, which should
    # save some string comparisons.
    info: HeaderInfo | CrossSection
    if (len(values) == 9 or len(values) == 11 or len(values) == 13) and values[1] == "Event":
        ##########################
        # Header v2 specification:
        ##########################
        # As of 22 June 2021, the formatting of the header is as follows:
        # This function was developed to parse it.
        # The header is defined as follows, with each entry separated by a `\t` character:
        # `# Event 1 weight 0.129547 EPangle 0.0116446 N_hadrons 236 (centrality 12.5) (pt_hat 47)`
        #  0 1     2 3      4        5       6         7         8    9          10     11     12
        # NOTE: pt_hat and centrality are optional (centrality added in Jan. 2025)
        #
        # The base length (i.e. with no optional values) is 9
        _base_line_length = 9

        # Extract optional values. They will be any beyond the base line length
        centrality, pt_hat = _parse_optional_header_values(
            optional_values=values[_base_line_length:],
        )

        # And store...
        info = HeaderInfo(
            event_number=int(values[2]),  # Event number
            event_plane_angle=float(values[6]),  # EP angle
            n_particles=int(values[8]),  # Number of particles
            event_weight=float(values[4]),  # Event weight
            centrality=centrality,  # centrality
            pt_hat=pt_hat,  # pt hat
        )
    elif len(values) == 5 and values[1] == "sigmaGen":
        # If we've hit the cross section, and we're not doing the initial extraction of the cross
        # section, this means that we're at the last line of the file, and should notify as such.
        # NOTE: By raising with the cross section, we make it possible to retrieve it, even though
        #       we've raised an exception here.
        raise parse_ascii_base.ReachedXSecAtEndOfFileException(parse_cross_section(line))
    else:
        _msg = f"Parsing of comment line failed: {values}"
        raise ValueError(_msg)

    return info


def _parse_header_line_format_v3(line: str) -> HeaderInfo:
    """Parse line that is expected to be a header according to the v3 file format.

    The most common case is that it's a header, in which case we parse the line. If it's not a header,
    we also check if it's a cross section, which we note as having found at the end of the file via
    an exception.

    Args:
        line: Line to be parsed.
    Returns:
        The HeaderInfo information that was extracted.
    Raises:
        ReachedXSecAtEndOfFileException: If we find the cross section.
    """
    # Parse the string.
    values = line.split("\t")
    # Compare by length first so we can short circuit immediately if it doesn't match, which should
    # save some string comparisons.
    info: HeaderInfo | CrossSection
    if (len(values) == 15 or len(values) == 17 or len(values) == 19) and values[1] == "Event":
        ##########################
        # Header v3 specification:
        ##########################
        # format including the vertex position and pt_hat
        # This function was developed to parse it.
        # The header is defined as follows, with each entry separated by a `\t` character:
        #  # Event 1 weight  1 EPangle 0 N_hadrons 169 vertex_x  0.6 vertex_y  -1.2  vertex_z  0 (centrality 12.5) (pt_hat  11.564096)
        #  0 1     2 3       4 5       6 7         8   9         10  11        12    13        14 15         16     17      18
        #
        # NOTE: pt_hat and centrality are optional (centrality added in Jan. 2025)
        #
        # The base length (i.e. with no optional values) is 15
        _base_line_length = 15

        # Extract optional values. They will be any beyond the base line length
        centrality, pt_hat = _parse_optional_header_values(
            optional_values=values[_base_line_length:],
        )

        # And store...
        info = HeaderInfo(
            event_number=int(values[2]),  # Event number
            event_plane_angle=float(values[6]),  # EP angle
            n_particles=int(values[8]),  # Number of particles
            event_weight=float(values[4]),  # Event weight
            vertex_x=float(values[10]),  # x vertex
            vertex_y=float(values[12]),  # y vertex
            vertex_z=float(values[14]),  # z vertex
            centrality=centrality,  # centrality
            pt_hat=pt_hat,  # pt hat
        )
    elif len(values) == 5 and values[1] == "sigmaGen":
        # If we've hit the cross section, and we're not doing the initial extraction of the cross
        # section, this means that we're at the last line of the file, and should notify as such.
        # NOTE: By raising with the cross section, we make it possible to retrieve it, even though
        #       we've raised an exception here.
        raise parse_ascii_base.ReachedXSecAtEndOfFileException(parse_cross_section(line))
    else:
        msg = f"Parsing of comment line failed: {values}"
        raise ValueError(msg)

    return info


# Register header parsing functions
_file_format_version_to_header_parser = {
    2: _parse_header_line_format_v2,
    3: _parse_header_line_format_v3,
    -1: _parse_header_line_format_unspecified,
}


def event_by_event_generator(
    f: Iterator[str], parse_header_line: Callable[[str], HeaderInfo]
) -> Iterator[HeaderInfo | str]:
    """Event-by-event generator using the JETSCAPE FinalState* model output file.

    Raises:
        ReachedXSecAtEndOfFileException: We've found the line with the xsec and error at the
            end of the file. Effectively, we've exhausted the iterator.
        ReachedEndOfFileException: We've hit the end of file without finding the xsec and
            error. This may be totally fine, depending on the version of the FinalState*
            output.
    """
    # Our first line should be the header, which will be denoted by a "#".
    # Let the calling know if there are no events left due to exhausting the iterator.
    try:
        header = parse_header_line(next(f))
        # logger.info(f"header: {header}")
        yield header
    except StopIteration:
        # logger.debug("Hit end of file exception!")
        raise parse_ascii_base.ReachedEndOfFileException() from None

    # From the header, we know how many particles we have in the event, so we can
    # immediately yield the next n_particles lines. And this will leave the generator
    # at the next event, or (if we've exhausted the iterator) at the end of file (either
    # at the xsec and error, or truly exhausted).
    yield from itertools.islice(f, header.n_particles)


def initialize_model_parameters(file_format_version: int) -> parse_ascii_base.ModelParameters:
    """Initialize model parameters and parsing functions for the JETSCAPE model.

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
    # 1. Identify the last line in the file, so we search for "\n". Note that there is special
    #    logic to avoid retrieving an end of line at the end of the file.
    # 2. Once we've scanned back far enough to find something, we'll ensure that we're
    #    looking at the right line by looking for the line containing "#\tsigmaGen".
    # NOTE(RJE): It would probably be enough to look for "#", but we used "\n" in the past and
    #            we know that it works, so better to just stick with it.
    extract_x_sec_func = functools.partial(
        parse_ascii_base.extract_x_sec_and_error,
        search_character_to_find_line_containing_cross_section="\n",
        start_of_line_containing_cross_section="#\tsigmaGen",
        parse_cross_section_line=parse_cross_section,
    )

    # Event-by-event generator
    # All we need is the parser for the header lines
    e_by_e_generator = functools.partial(
        event_by_event_generator,
        parse_header_line=_file_format_version_to_header_parser[file_format_version],
    )

    return parse_ascii_base.ModelParameters(
        model_name="jetscape",
        column_names=["particle_index", "particle_ID", "status", "E", "px", "py", "pz", "eta", "phi"],
        extract_x_sec_and_error=extract_x_sec_func,
        event_by_event_generator=e_by_e_generator,
    )
