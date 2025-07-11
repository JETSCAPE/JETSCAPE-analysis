"""Skim a large ascii output file and split into smaller compressed files.

.. codeauthor:: James Mulligan <james.mulligan@berkeley.edu>, LBL/UCB
.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jetscape_analysis.analysis.reader import parse_ascii
from jetscape_analysis.base import common_base, helpers


################################################################
class SkimAscii(common_base.CommonBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, input_file: Path, output_dir: Path, events_per_chunk: int = 50000, **kwargs):
        super().__init__(**kwargs)
        self.input_file = input_file
        self.output_dir = output_dir

        self.event_id: int = 0
        self.events_per_chunk = events_per_chunk

    # ---------------------------------------------------------------
    # Main processing function for a single pt-hat bin
    # ---------------------------------------------------------------
    def skim(self) -> None:
        # The parser writes out a parquet file containing the ascii events,
        # reformatted and compressed.
        parse_ascii.parse_to_parquet(
            base_output_filename=self.output_dir, input_filename=self.input_file, events_per_chunk=self.events_per_chunk
        )


##################################################################
if __name__ == "__main__":
    # Setup
    helpers.setup_logging(level=logging.INFO)

    # Define arguments
    parser = argparse.ArgumentParser(description="Convert (skim) ascii output files to parquet")
    parser.add_argument(
        "-i",
        "--inputFile",
        action="store",
        type=Path,
        metavar="inputDir",
        default=Path("/home/jetscape-user/JETSCAPE-analysis/test.out"),
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        action="store",
        type=Path,
        metavar="outputDir",
        default=Path("/home/jetscape-user/JETSCAPE-analysis/TestOutput.parquet"),
        help="Output directory and filename template for output to be written to",
    )
    parser.add_argument(
        "-n",
        "--nEventsPerFile",
        action="store",
        type=int,
        metavar="nEventsPerFile",
        default=50000,
        help="Number of events to store in each parquet file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid inputDir is given, exit
    if not args.inputFile.exists():
        _msg = f'File "{args.inputFile}" does not exist! Exiting!'
        raise RuntimeError(_msg)

    analysis = SkimAscii(input_file=args.inputFile, output_dir=args.outputDir, events_per_chunk=args.nEventsPerFile)
    analysis.skim()
