#!/usr/bin/env python3

""" Base class to analyze a JETSCAPE output file

You should create a user class that inherits from this one. See analyze_events_STAT.py for an example.

The outputdir should contain a JETSCAPE output file in parquet format

See README for pre-requisites.

.. codeauthor:: James Mulligan <james.mulligan@berkeley.edu>, UC Berkeley
"""

from __future__ import annotations

# General
import os
import sys
import yaml
import time
from pathlib import Path
from numba import jit

# Analysis
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

from jetscape_analysis.base import common_base

################################################################
class AnalyzeJetscapeEvents_BaseSTAT(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file="", input_file="", output_dir="", **kwargs):
        super(AnalyzeJetscapeEvents_BaseSTAT, self).__init__(**kwargs)

        self.config_file = config_file
        self.input_file_hadrons = input_file
        self.output_dir = Path(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            if 'n_event_max' in config:
                self.n_event_max = config['n_event_max']
            else:
                self.n_event_max = -1

            self.thermal_rejection_fraction = config.get('thermal_rejection_fraction', 0.)

        # Check whether pp or AA
        if 'PbPb' in self.input_file_hadrons or 'AuAu' in self.input_file_hadrons:
            self.is_AA = True
        else:
            self.is_AA = False

        # If AA, get centrality bin
        self.use_event_based_centrality = False
        if self.is_AA:
            _final_state_hadrons_path = Path(self.input_file_hadrons)
            # For an example filename of "jetscape_PbPb_Run0005_5020_0001_final_state_hadrons_00.parquet",
            # - the run number is index 2
            _job_identifier = _final_state_hadrons_path.stem.split("_")[2]
            if "Run" in _job_identifier:
                # We're using a standard production with a run number - look for the run info file.
                _run_number = _job_identifier
                # - the file index is at index 4 (in the example, it extracts `1` as an int)
                _file_index = int(_final_state_hadrons_path.name.split('_')[4])
                run_info_path = _final_state_hadrons_path.parent / f"{_run_number}_info.yaml"
                with open(run_info_path, 'r') as f:
                    _run_info = yaml.safe_load(f)
                    centrality_string = _run_info["index_to_hydro_event"][_file_index].split('/')[0].split('_')
                    # index of 1 and 2 based on an example entry of "cent_00_01"
                    self.centrality = [int(centrality_string[1]), int(centrality_string[2])]
            else:
                # No run info available - need to retrieve the centrality event-by-event
                self.use_event_based_centrality = True

        # If AA, initialize constituent subtractor
        self.constituent_subtractor = None
        if self.is_AA:
            print('Constituent subtractor is enabled.')
            constituent_subtractor = config['constituent_subtractor']
            max_distance = constituent_subtractor['R_max']
            max_eta = constituent_subtractor['max_eta']
            ghost_area = constituent_subtractor['ghost_area']
            bge_rho_grid_size = constituent_subtractor['bge_rho_grid_size']
            self.bge_rho = fj.GridMedianBackgroundEstimator(max_eta, bge_rho_grid_size)
            self.constituent_subtractor = fjcontrib.ConstituentSubtractor()
            self.constituent_subtractor.set_background_estimator(self.bge_rho)
            self.constituent_subtractor.set_max_distance(max_distance)
            self.constituent_subtractor.set_ghost_area(ghost_area)
            self.constituent_subtractor.set_max_eta(max_eta)
            self.constituent_subtractor.initialize()
            print(dir(self.constituent_subtractor))
        else:
            print('Constituent subtractor is disabled.')

    # ---------------------------------------------------------------
    # Main processing function
    # ---------------------------------------------------------------
    def analyze_jetscape_events(self):

        print('Analyzing events ...')

        # Initialize output objects
        self.initialize_output_objects()

        # Read chunk of events into a dataframe
        # Fields: particle_ID, status, E, px, py, pz
        df_event_chunk = pd.read_parquet(self.input_file_hadrons)
        if self.n_event_max < 0:
            self.n_event_max = df_event_chunk.shape[0]

        # Iterate through events
        self.analyze_event_chunk(df_event_chunk)

        # Write analysis task output to ROOT file
        self.write_output_objects()

        print('Done!')

    # ---------------------------------------------------------------
    # Analyze event chunk
    # ---------------------------------------------------------------
    def analyze_event_chunk(self, df_event_chunk):

        # Loop through events
        start = time.time()
        weight_sum = 0.
        # Track the overall centrality range
        centrality_range_min, centrality_range_max = 100, 0
        for i, event in df_event_chunk.iterrows():

            if i % 1000 == 0:
                print(f'event: {i}    (time elapsed: {time.time() - start} s)')

            if i > self.n_event_max:
                break

            # Store dictionary of all observables for the event
            self.observable_dict_event = {}

            # Update self.centrality dynamically per event
            if self.is_AA:
                if self.use_event_based_centrality:
                    # Double check that the centrality is available in the event dictionary. If not, need to raise the issue early.
                    if i == 0 and "centrality" not in event:
                        msg = "Running AA, there is no run info file, and event-by-event centrality is not available, so we are unable to proceed. Please check configuration"
                        raise ValueError(msg)
                    self.centrality = [int(np.floor(event['centrality'])), int(np.ceil(event['centrality']))]  # Dynamically set centrality; values are passed from the parquet file
                else:
                    self.centrality = self.default_centrality  # Use fixed centrality; values are passed from the Run_info.yaml file

            # Call user-defined function to analyze event
            self.analyze_event(event)

            # Fill the observables dict to a new entry in the event list
            event_weight = event['event_weight']
            weight_sum += event_weight
            if self.event_has_entries(self.observable_dict_event):

                # Fill event cross-section weight
                self.observable_dict_event['event_weight'] = event_weight
                self.observable_dict_event['pt_hat'] = event['pt_hat']

                # Add event-wise centrality (same for all events in pre-computed hydro; varies event-by-event for real_time_hydro)
                if self.is_AA:
                    self.observable_dict_event['centrality_min'] = self.centrality[0]
                    self.observable_dict_event['centrality_max'] = self.centrality[1]
                    # This is trivially the same for each event for the pre-computed hydro,
                    # but it varies for the on-the-fly case.
                    centrality_range_min = min(self.centrality[0], centrality_range_min)
                    centrality_range_max = max(self.centrality[1], centrality_range_max)

                self.output_event_list.append(self.observable_dict_event)

        # Get total cross-section (same for all events at this point), weight sum, and centrality
        self.cross_section_dict['cross_section'] = event['cross_section']
        self.cross_section_dict['cross_section_error'] = event['cross_section_error']
        self.cross_section_dict['n_events'] = self.n_event_max
        self.cross_section_dict['weight_sum'] = weight_sum
        if self.is_AA:
            self.cross_section_dict['centrality_range_min'] = int(np.floor(centrality_range_min))
            self.cross_section_dict['centrality_range_max'] = int(np.ceil(centrality_range_max))

    # ---------------------------------------------------------------
    # Initialize output objects
    # ---------------------------------------------------------------
    def initialize_output_objects(self):

        # Initialize list to store observables
        # Each entry in the list stores a dict for a given event
        self.output_event_list = []

        # Store also the total cross-section (one number per file)
        self.cross_section_dict = {}

    # ---------------------------------------------------------------
    # Save output event list into a dataframe
    # ---------------------------------------------------------------
    def event_has_entries(self, event_dict):

        return bool([obs for obs in event_dict.values() if obs != []])

    # ---------------------------------------------------------------
    # Check if event centrality is within observable's centrality
    # ---------------------------------------------------------------
    def centrality_accepted(self, observable_centrality_list):

        # AA
        if self.is_AA:

            for observable_centrality in observable_centrality_list:
                if self.centrality[0] >= observable_centrality[0]:
                    if self.centrality[1] <= observable_centrality[1]:
                        return True
            return False

        # pp
        else:
            return True

    # ---------------------------------------------------------------
    # Save output event list into a dataframe
    # ---------------------------------------------------------------
    def write_output_objects(self):

        # Convert to pandas, and then arrow.
        self.output_dataframe = pd.DataFrame(self.output_event_list)
        #self.output_dataframe = ak.Array(self.output_event_list)
        table = pa.Table.from_pandas(self.output_dataframe)

        # Write to parquet
        # Determine the types for improved compression when writing
        # See writing to parquet in the final state hadrons parser for more info.
        float_types = [np.float32, np.float64]
        float_columns = list(self.output_dataframe.select_dtypes(include=float_types).keys())
        other_columns = list(self.output_dataframe.select_dtypes(exclude=float_types).keys())
        # NOTE: As of 27 April 2021, this doesn't really work right because too many columns
        #       are of the "object" type. We may need to revise the output format to optimize
        #       the output size.
        print(f"float_columns: {float_columns}")
        print(f"other_columns: {other_columns}")
        pq.write_table(
            table, self.output_dir / self.output_file, compression="zstd",
            use_dictionary=other_columns,
            use_byte_stream_split=float_columns,
        )

        # Write cross-section to separate file
        cross_section_dataframe = pd.DataFrame(self.cross_section_dict, index=[0])
        cross_section_table = pa.Table.from_pandas(cross_section_dataframe)
        filename = self.output_file.replace('observables', 'cross_section')
        pq.write_table(cross_section_table, self.output_dir / filename, compression="zstd")

    # ---------------------------------------------------------------
    # Fill hadrons into vector of fastjet pseudojets
    #
    # By default, select all particles
    # If select_status='+', select only positive status particles
    # If select_status='-', select only negative status particles
    #
    # We return the list of fastjet::PseudoJets, where the user_index is set to:
    #   user_index = (+/-)i,
    #   where i is the index in the list, and is weighted by (+/-) for positive/negative status particles
    # We also return the list of PID values, so that it can later be determined from the index i
    # ---------------------------------------------------------------
    def fill_fastjet_constituents(self, event, select_status=None, select_charged=False):

        # Construct indices according to particle status
        if select_status == '-':
            status_mask = (event['status'] < 0)
        elif select_status == '+':
            status_mask = (event['status'] > -1)
        else:
            # Picked a value to make an all true mask. We don't select anything
            status_mask = event['status'] > -1e6

        # Construct indices according to charge
        charged_mask = get_charged_mask(event['particle_ID'], select_charged)

        # Get selected particles
        full_mask = status_mask & charged_mask
        px = event['px'][full_mask]
        py = event['py'][full_mask]
        pz = event['pz'][full_mask]
        e = event['E'][full_mask]
        pid = event['particle_ID'][full_mask]

        # Define status_factor -- either +1 (positive status) or -1 (negative status)
        status_selected = event['status'][full_mask] # Either 0 (positive) or -1 (negative)
        status_factor = 2*status_selected + 1 # Change to +1 (positive) or -1 (negative)
        # NOTE: Need to explicitly convert to np.int8 -> np.int32 so that the status factor can be
        #       set properly below. Otherwise, it will overflow when setting the user index if there
        #       are too many particles.
        status_factor = status_factor.astype(np.int32)
        for status in np.unique(status_selected): # Check that we only encounter expected statuses
            if status not in [0,-1]:
                sys.exit(f'ERROR: fill_fastjet_constituents -- unexpected particle status -- {status}')

        # Create a vector of fastjet::PseudoJets from arrays of px,py,pz,e
        fj_particles = fjext.vectorize_px_py_pz_e(px, py, pz, e)

        # Set user_index = (+/-)(i+1), so that we encode both the status information and the pid index
        # Note that we use i+1 since 0-index otherwise does not distinguish +/-
        # We then have: pid_index = abs(user_index) - 1
        # In this way, user_index > 0 corresponds to positive status particles, and user_index < 0 corresponds to negative status particles
        if len(fj_particles) == len(status_factor):
            [fj_particles[i].set_user_index(int(status_factor[i]*(i+1))) for i,_ in enumerate(fj_particles)]
        else:
            sys.exit(f'ERROR: fill_fastjet_constituents -- len(fj_particles) != {len(status_factor)} -- {len(fj_particles)} vs. {len(status_factor)}')

        return fj_particles, pid
    def is_prompt_photon(self, photon):
        # TODO we currently do not have the informmation to handle this properly 
        # since the information is missing in the MC. Return True for now.
        return True
    # find out if the particle is isolated or not
    def is_isolated(self, trigger_particle, iso_particles_charged_pos,iso_particles_charged_neg,iso_R, iso_Et_max):
        """Find out if the particle is isolated or not.
        
        Args:
            event: Event containing particle information
            trigger_particle: fastjet::PseudoJet of the particle to check isolation for
            iso_particles_charged: list of fastjet::PseudoJets for the charged particles to consider for calculation of particles in cone
            iso_particles_neutral: list of fastjet::PseudoJets for the neutral particles to consider for calculation of particles in cone
            iso_R: radius of the isolation cone
            iso_Et_max: maximum sum pt of particles in cone in order for trigger_particle to be considered isolated
            isolation_type: type of isolation to use, either 'full' or 'charged' or 'neutral'
        Returns:
            bool: True if the particle is isolated, False otherwise
        """
        # Calculate sum Et of particles in cone around trigger particle
        sum_Et = 0.
        for particle in iso_particles_charged_pos:
            if trigger_particle.delta_R(particle) < iso_R:
                # Skip the trigger particle itself
                if particle.user_index() == trigger_particle.user_index():
                    continue
                sum_Et += particle.Et()
        for particle in iso_particles_charged_neg:
            if trigger_particle.delta_R(particle) < iso_R:
                # Skip the trigger particle itself
                if particle.user_index() == trigger_particle.user_index():
                    continue
                sum_Et -= particle.Et() # subtract holes
        # Return whether sum Et is below threshold
        return sum_Et < iso_Et_max
    # TODO implement this after asking Peter
    def build_trigger_response_matrix_STAR(self):
        # create a 2D matrix from 6 to 30 GeV/c in bine of 1 GeV
        Et_part_bins = np.arange(6, 30, 1)
        Et_det_bins = np.arange(9, 20, 1)
        # create a 2D matrix of size Et_bins x Et_bins
        trigger_response_matrix = np.zeros((len(Et_det_bins), len(Et_part_bins)))
        # fill the matrix with the trigger response
        # For each Et_det bin, calculate probability distribution for Et_part
        TES = 1
        TER = 1
        for i, Et_det in enumerate(Et_det_bins):
            # For each Et_part bin, calculate probability that this Et_part gives the Et_det
            if Et_det > 9 and Et_det <= 11:
                TES = 0.9797
                TER = 0.0812
            elif Et_det > 11 and Et_det <= 15:
                TES = 0.9777
                TER = 0.0783
            elif Et_det > 15 and Et_det <= 20:
                TES = 0.9774
                TER = 0.0756
            for j, Et_part in enumerate(Et_part_bins):
                # Probability is given by Gaussian with:
                # mean = TES * Et_part (expected detector response)
                # sigma = TER * Et_part (detector resolution)
                mean = TES * Et_part
                sigma = TER * Et_part
                
                # Calculate probability that this Et_part gives the Et_det bin
                # Using error function (erf) for Gaussian integral over bin width
                bin_low = Et_det - 0.5  # Lower edge of Et_det bin
                bin_high = Et_det + 0.5 # Upper edge of Et_det bin
                
                prob = 0.5 * (scipy.special.erf((bin_high - mean)/(sigma * np.sqrt(2))) - 
                             scipy.special.erf((bin_low - mean)/(sigma * np.sqrt(2))))
                
                trigger_response_matrix[i,j] = prob
                
        # Normalize each Et_det row to sum to 1
        row_sums = trigger_response_matrix.sum(axis=1)
        trigger_response_matrix = trigger_response_matrix / row_sums[:, np.newaxis]
        return 

    # ---------------------------------------------------------------
    # Function to find all final state photons in event
    # ---------------------------------------------------------------
    def fill_photons(self, event):
        """Find all final state photons in the event and return as fastjet particles.
        
        Args:
            event: Event containing particle information
            
        Returns:
            tuple: (fj_photons, photon_indices) where:
                - fj_photons is list of fastjet::PseudoJets for photons
                - photon_indices are the indices of the photons in the original event
        """
        # Select photons (PID = 22) with positive status (TODO ask raymond what status exactly means)
        # TODO check with Raymond how to handle holes 
        photon_mask = (event['particle_ID'] == 22) & (event['status'] > -1)
        
        # Get photon kinematics
        px = event['px'][photon_mask]
        py = event['py'][photon_mask]
        pz = event['pz'][photon_mask]
        e = event['E'][photon_mask]
        
        # Create fastjet particles
        fj_photons = fjext.vectorize_px_py_pz_e(px, py, pz, e)
            
        return fj_photons
    # Function to obtain jet_pt and jet_pt_uncorrected according to jet collection
    def get_jet_pt(self,jet,jetR,hadrons_negative,jet_collection_label=''):
        jet_pt, jet_pt_uncorrected = 0., 0.
        holes_in_jet = []
        if jet_collection_label in ['_shower_recoil', '_negative_recombiner']:
            for hadron in hadrons_negative:
                if jet.delta_R(hadron) < jetR:
                    holes_in_jet.append(hadron)
        # Correct the pt of the jet, if applicable
        # For pp or negative recombiner or constituent subtraction case, we do not need to adjust the pt
        # For the shower+recoil case, we need to subtract the hole pt
        if jet_collection_label in ['', '_negative_recombiner', '_constituent_subtraction']:
                jet_pt = jet_pt_uncorrected = jet.pt()
        elif jet_collection_label in ['_shower_recoil']:
                negative_pt = 0.
                for hadron in holes_in_jet:
                    negative_pt += hadron.pt()
                jet_pt_uncorrected = jet.pt()               # uncorrected pt: shower+recoil
                jet_pt = jet_pt_uncorrected - negative_pt   # corrected pt: shower+recoil-holes
        return jet_pt, jet_pt_uncorrected


    # ---------------------------------------------------------------
    # This function is called once per event
    # You must implement this
    # ---------------------------------------------------------------
    def analyze_event(self, event):
        raise NotImplementedError('You must implement analyze_event()!')

# ---------------------------------------------------------------
# Construct charged particle mask
# ---------------------------------------------------------------
@jit(nopython=True)
def get_charged_mask(pid, select_charged: bool):
    """ Create mask for selected a set of charged particles based on PID.

    Note:
        This function assumes that the same set of charged particles are selected
        for all charged-particle jets (ie. ALICE and STAR). Although the charged
        particle selections for some of the hadron observables vary between experiments,
        this seems like

    Args:
        pid: PID values associated with the charged particles in an event.
        select_charged: If True, actually select charged particles. If False,
            just return an all True mask (if for full jets).

    Returns:
        Mask selecting the particles.
    """
    # Default to an all true mask
    charged_mask = np.ones(len(pid)) > 0
    if select_charged:
        # Create an all false mask. We'll fill it in with the charged constituents
        charged_mask = np.ones(len(pid)) < 0
        for i, pid_value in enumerate(pid):
            # (e-, mu-, pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
            if np.abs(pid_value) in [11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334]:
                charged_mask[i] = True

    return charged_mask


@jit(nopython=True)  # type: ignore
def dphi_in_range_for_hadron_correlations(dphi: float, min_phi: float = -np.pi / 2, max_phi: float = 3 * np.pi / 2) -> float:
    """ Put dphi in range min_phi <= dphi < max_phi

    Args:
        dphi: phi value to normalize.
        min_phi: minimum allowed phi. Default: -pi/2
        max_phi: maximum allowed phi. Default: 3*pi/2
    Returns:
        Normalized phi
    """
    if dphi < min_phi:
        dphi += 2 * np.pi
    elif dphi >= max_phi:
        dphi -= 2 * np.pi
    return dphi
