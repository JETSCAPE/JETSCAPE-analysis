"""Ascii reader class

.. codeauthor :: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from jetscape_analysis.analysis.event import event_ascii
from jetscape_analysis.analysis.reader import reader_base


################################################################
class ReaderAscii(reader_base.ReaderBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, input_file_hadrons="", input_file_partons="", **kwargs):
        super().__init__(**kwargs)

        self.event_list_hadrons = self.parse_event(input_file_hadrons)
        self.current_event = 0
        self.n_events = len(self.event_list_hadrons)

        if os.path.exists(input_file_partons):
            self.event_list_partons = self.parse_event(input_file_partons)
        else:
            self.event_list_partons = None

        if os.path.exists(input_file_hadrons) and os.path.exists(input_file_partons):
            if len(self.event_list_hadrons) != len(self.event_list_partons):
                sys.exit(
                    f"Final state partons has {len(self.event_list_hadrons)} events, but partons has {len(self.event_list_partons)}."
                )

    # ---------------------------------------------------------------
    # Parse the file into a list of events, each consisting of a list of lines
    # (applied separately for final-state hadrons and partons)
    # ---------------------------------------------------------------
    def parse_event(self, input_file):
        event_list = []
        event = None
        with open(input_file) as f:
            for line in f.readlines():
                # If a new event, write the previous event and then clear it
                if line.startswith("#"):
                    if event:
                        event_list.append(event)
                    event = []

                else:
                    event.append(np.array(line.rstrip("\n").split(), dtype=float))

            # Write the last event
            event_list.append(event)

        return event_list

    # ---------------------------------------------------------------
    # Get next event
    # Return event if successful, False if unsuccessful
    # ---------------------------------------------------------------
    def next_event(self):
        if self.current_event < self.n_events:
            self.current_event += 1
            event_hadrons = self.event_list_hadrons[self.current_event - 1]
            if self.event_list_partons:
                event_partons = self.event_list_partons[self.current_event - 1]
            else:
                event_partons = ""
            return event_ascii.EventAscii(event_hadrons, event_partons)
        sys.exit(f"Current event {self.current_event} greater than total n_events {self.n_events}")
