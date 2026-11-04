"""Ascii reader class

.. codeauthor:: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import annotations

import sys

from jetscape_analysis.analysis.event import event_ascii
from jetscape_analysis.analysis.reader import reader_base


################################################################
class ReaderAsciiParsed(reader_base.ReaderBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, event_chunk_hadrons=None, event_chunk_partons=None, **kwargs):
        super().__init__(**kwargs)

        self.event_chunk_hadrons = event_chunk_hadrons
        self.event_chunk_partons = event_chunk_partons

        self.current_event = 0
        self.n_events = len(self.event_chunk_hadrons)

        if self.event_chunk_partons:
            if len(self.event_chunk_hadrons) != len(self.event_chunk_partons):
                sys.exit(
                    f"Final state partons has {len(self.event_list_hadrons)} events, but partons has {len(self.event_list_partons)}."
                )

    # ---------------------------------------------------------------
    # Get next event
    # Return event if successful, False if unsuccessful
    # ---------------------------------------------------------------
    def next_event(self):
        if self.current_event < self.n_events:
            self.current_event += 1
            event_hadrons = self.event_chunk_hadrons[self.current_event - 1]
            if self.event_chunk_partons:
                event_partons = self.event_chunk_partons[self.current_event - 1]
            else:
                event_partons = None
            return event_ascii.EventAscii(event_hadrons, event_partons)
        sys.exit(f"Current event {self.current_event} greater than total n_events {self.n_events}")
