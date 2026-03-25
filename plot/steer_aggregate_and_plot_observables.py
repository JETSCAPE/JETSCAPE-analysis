"""
  Macro to steer calculation of observables from histograms produced from a set of runs on XSEDE and uploaded to OSN.

  (See steer_plot_observables.py instead for details on the full workflow of computing observables from final_state_hadrons).

  To run this script:

    - Edit the configuration options at the start of main()

    - Set up environment:
        If downloading from OSN ("download_runinfo" or "download_histograms" True):
            cd STAT-XSEDE-2021/scripts
            python3 -m venv venv            # only needed the first time
            source venv/bin/activate
            pip install .                  # only needed the first time
        If merging/plotting histograms ("merge_histograms", "aggregate_histograms", or "plot_histograms" True),
        need ROOT compiled with python, e.g.:
            [enter virtual environment with needed python3 packages, and same python3 version as ROOT build]
            export ROOTSYS=/home/software/users/james/heppy/external/root/root-current
            source $ROOTSYS/bin/thisroot.sh

    - Run script: python3 plot/steer_aggregate_and_plot_observables.py

------------------------------------------------------------------------

  The workflow is as follows, with each step toggle-able below:

   (1) Download run info for the set of runs on OSN.

       This involves two pieces:
         (i) Download runs.yaml for each facility, from STAT-XSEDE-2021/docs/DataManagement
         (ii) Using these runs.yaml, download run_info.yaml for all runs and populate a dictionary with all relevant info for each run

       By default this will only download files that you have not downloaded locally. You can force re-download all with force_download=True.

   (2) Download histograms for each run.

       By default this will only download files that you have not downloaded locally. You can force re-download all with force_download=True.

   (3) Merge histograms for each run together.

   (4) Aggregate runs, using the dictionary from step (1).

       For each (sqrts, system, parametrization_type, design_point_index),
       merge all histograms into a single one, summed over facilities, run numbers, centralities.

   (5) Plot final observables for each design point, and write table for input to Bayesian analysis.

       In the AA case, we plot the AA/pp ratios
       In the pp case, we plot the pp distributions

  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import logging
import os
import pathlib
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import ROOT
except ImportError:
    pass

# Suppress performance warning which seems to come from h5py
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


logger = logging.getLogger(__name__)


# Configuration options
# -----------------------------------------------------------------
# Set which options you want to execute
# To edit the set of observables that are plotted, see the self.analysis variable in plot_results_STAT.py
# Note: all observables that are plotted are saved to hdf5, and are then written to tables
# Note: plotting script may have warnings about missing histograms, this is usually due to some missing centrality bins for certain design points
analysis_name = "Analysis1"
model_name = "jetscape"
# analysis_name = 'hybrid_production_0'
# model_name = "hybrid"
download_runinfo = True  # if true, download all run_info.yaml files specified in DataManagement (runs.yaml)
download_histograms = False  # if true, download all histogram files from OSN
list_paths_for_selected_design_points = False
merge_histograms = True  # if true, merge all histograms for each run into a single file
aggregate_histograms = False  # if true, aggregate all histograms for each design point into a single file
plot_and_save_histograms = False  # if true, plot and save all histograms
write_tables = False  # if true, write tables for input to Bayesian analysis
plot_global_QA = False  # if true, plot global QA

# re-analysis parameters
# -----------------------------------------------------------------
download_final_state_hadrons = False  # if False, final_state_hadrons.parquet files for local analysis
analysis_final_state_hadrons = False  # if True, analyze final_state_hadrons.parquet files locally
local_analysis_facility = (
    "test_587cluster"  # facility to run local re-analysis as defined in cluster config of STAT-XSEDE-2021
)

# version number used to identify the version of the analysis. Histograms and observables are stored on OSN in facility/RunX/histograms/versionY
# if version -1 is specified, histogram files for download will be downloaded from facility/RunX/histograms without version specifier
# if version -1 is specified, no upload of the results from reanalysis is possible
analysis_version = -1
force_reanalysis = False  # force to analyse and download again, even if histogram files are present
delete_local_final_state_hadrons_after_analysis = True  # delete final state hadron files after analysis is done
randomize_run_order = (
    True  # if true, runs will be shuffled into random order. This should be on by default, especially for benchmarking
)

# Processing options
force_download = False
download_threads = 5  # number of threads to download from OSN
n_cores = 20  # only used for merging with hadd
n_tasks = 50  # number of analysis tasks to be submitted to local_analysis_facility
max_n_slurm_jobs = 50
skip_run_binomial = False

# Location where histogram files and final state hadron files are stored on OSN
# -----------------------------------------------------------------
facilities = ["bridges2", "expanse"]
# facilities = ["cambridge"]

# Settings related to software and data locations
# -----------------------------------------------------------------
stat_xsede_2021_dir = Path("/jetscapeOpt/stat-xsede-2021")
jetscape_analysis_dir = Path("/jetscapeOpt/jetscape-analysis")
local_base_output_dir = Path("/rstorage/rehlers/hybrid-bayesian/aggregation/production_0/2026-03-19")
analysis_container_path = "/software/flo/myJETSCAPE/STAT-XSEDE-2021/containers/stat_local_gcc_v3.6.sif"
# This is where logbook used for bookkeeping is stored.
# NOTE: You can customize this to another directory - `stat-xsede-2021/docs/DataManagement` is just the default location
base_bookkeeping_path = Path(stat_xsede_2021_dir) / "docs" / "DataManagement"

# re-analysis debug options
# -----------------------------------------------------------------
do_debug_same_run = False  # if true, only process the same run over and over, useful for debugging

# debug option that allows to pick only a specific set of design points, cme, and centrality
# this is useful for testing a specific design point
# if nothing special should be selected, set to empty list. To select multiple options, separate by commas
debug_design_points = []
debug_parametrization_type = []
debug_sqrts = []
debug_centrality = []
debug_calculation_type = []
debug_system = []

# Example
# debug_design_points = [91]
# debug_parametrization_type = ['exponential']
# debug_sqrts = [5020]
# debug_centrality = [[0,10]]
# debug_calculation_type = ['jet_energy_loss']
# debug_system = ['PbPb']

# No editing needed beyond this point
# -----------------------------------------------------------------
if analysis_version != -1:
    local_base_output_dir = local_base_output_dir / f"version{analysis_version}"
final_state_hadrons_download_location = local_base_output_dir / "final_state_hadrons_per_run"
reananalysis_output_location_base = local_base_output_dir / "histograms_per_run"


def download_server(facilities, runs, buffer_size=5):
    """Download server for retrieving final state hadron files from remote facilities.

    Downloads files from specified facilities/runs into a local buffer directory. The program keeps
    downloading file until the buffer is full and waits 60s before rechecking.
    The main analysis thread will delete the files from the buffer once they are processed.
    A DONE file is created in the buffer directory once the download for a run is finished.
    The main analysis thread will check if the DONE file is present and start the re-analysis.

    Args:
        facilities (list): List of facility names to download from (e.g. ['bridges2', 'expanse'])
        runs (dict): Dictionary mapping facility names to lists of run IDs
        buffer_size (int, optional): Maximum number of files to keep in download buffer. Defaults to 5.
    """
    print("[Downloader] Starting download server...")  # noqa: T201
    for facility in facilities:
        for run in runs[facility]:
            # use for debugging only. Continue to take the same run for testing
            if do_debug_same_run:
                run = runs[facility][0]  # noqa: PLW2901
            run_number = int(run[3:])

            # If directory exists locally, check number of histograms already downloaded and number on OSN
            final_state_hadron_dir = final_state_hadrons_download_location / f"{facility}/{run}"
            reananalysis_output_location = reananalysis_output_location_base / f"{facility}/{run}"

            # before we download any analysis output, check if we already have analysis output that we want. If it is already there then skip this run
            if analysis_final_state_hadrons and not force_reanalysis and not do_debug_same_run:
                histpath = reananalysis_output_location / "histograms"
                # before starting analysis check if output directory already exists
                if histpath.exists():
                    # check number of output files
                    n_histoutput = len([h for h in histpath.iterdir() if h.suffix == ".root"])
                    n_parquet_local = len([h for h in final_state_hadron_dir.iterdir() if h.suffix == ".parquet"])

                    if n_histoutput == n_parquet_local:
                        print(  # noqa: T201
                            f"[Downloader] Analysis already done for run {run_number}, I found {n_histoutput} histograms, will not download ..."
                        )
                        continue

            if final_state_hadron_dir.exists():
                n_parquet_local = len([h for h in final_state_hadron_dir.iterdir() if h.suffix == ".parquet"])

                script = stat_xsede_2021_dir / "scripts/js_stat_xsede_steer/count_files_on_OSN.py"
                cmd = f"python3 {script!s} -s {facility}/{run}/ -f final_state_hadrons"
                proc = subprocess.run(cmd, check=False, shell=True, stdout=subprocess.PIPE)
                output = proc.stdout
                n_parquet_expected = int(output.split()[-1])

            # Download all files we are missing (or force download, if requested)
            if final_state_hadron_dir.exists() and n_parquet_expected == n_parquet_local and not force_download:
                print(  # noqa: T201
                    f"Hadron file dir ({final_state_hadron_dir}) already exists and n_parquet_expected ({n_parquet_expected}) == n_parquet_local ({n_parquet_local}), will not re-download"
                )
                print()  # noqa: T201
            else:
                if force_download or do_debug_same_run:
                    print("Force download enabled -- re-download all parquet files...")  # noqa: T201
                if final_state_hadron_dir.exists() and n_parquet_expected != n_parquet_local:
                    print(  # noqa: T201
                        f"Histogram dir already exists, but n_parquet_expected ({n_parquet_expected}) does not equal n_parquet_local ({n_parquet_local}), so we will redownload."
                    )

                # check in final_state_hadrons_download_location recursively for all files called DONE
                # if there are more than buffer_size files, wait until there are less than buffer_size files
                # then download the next file

                while True:
                    # check if there are too many files in the buffer
                    bufferedRuns = 0
                    for _, _, files in os.walk(final_state_hadrons_download_location):
                        for file in files:
                            if file == "DONE":
                                bufferedRuns += 1
                    if bufferedRuns >= buffer_size:
                        print("[Downloader] Buffer full, waiting 60s before rechecking...")  # noqa: T201
                        time.sleep(60)
                    else:
                        break
                # Create the directory if it does not exists
                final_state_hadron_dir.mkdir(exist_ok=True, parents=True)
                print(f"[Downloader] Downloading run {run_number} ...")  # noqa: T201
                download_script = stat_xsede_2021_dir / "scripts/js_stat_xsede_steer/download_from_OSN.py"
                cmd = f"python3 {download_script!s} -s {facility}/{run}/ -d {final_state_hadrons_download_location!s} -f final_state_hadrons -c {download_threads}"
                subprocess.run(cmd, check=True, shell=True)
                # create a file DONE in the directory to signal that the download is finished
                (final_state_hadron_dir / "DONE").touch()
                print()  # noqa: T201


def main():
    # -----------------------------------------------------------------
    # (1) Download info for all runs from OSN, and create a dictionary with all info needed to download and aggregate histograms
    # -----------------------------------------------------------------
    if (
        download_runinfo
        or download_histograms
        or list_paths_for_selected_design_points
        or merge_histograms
        or aggregate_histograms
        or download_final_state_hadrons
    ):
        runs = {}
        run_dictionary = {}
        missing_runinfo = defaultdict(list)
        skipped_runs = defaultdict(list)

        # (i) Load the runs.yaml for each facility from STAT-XSEDE-2021
        for facility in facilities.copy():
            runs_filename = base_bookkeeping_path / f"{analysis_name}/{facility}/runs.yaml"
            logger.info(f"Searching for runs.yaml in: {runs_filename}")
            if runs_filename.exists():
                with runs_filename.open() as f:
                    runs[facility] = [str(name) for name in list(yaml.safe_load(f).keys())]
            else:
                logger.warning(f"runs.yaml not found for {facility}. Removing it from the facilities list.")
                facilities.remove(facility)

        # (ii) Using these runs.yaml, download run_info.yaml for all runs and populate dictionary with relevant info for aggregation
        stat_xsede_available = False
        try:
            from js_stat_xsede_steer import file_management
            stat_xsede_available = True
        except ImportError:
            msg = "stat-xsede is not available, so some functionality is not available."
            logger.warning(msg)

        for facility in facilities:
            # if requested, shuffle runs into random order
            if randomize_run_order:
                random.shuffle(runs[facility])

            run_dictionary[facility] = defaultdict(dict)
            process_list = []

            # make a list of tuples that contain source and target
            runinfo_pairs = []

            for i, run in enumerate(runs[facility].copy()):
                run_info_download_location = local_base_output_dir / "run_info"
                run_info_file = run_info_download_location / f"{facility}/{run}/{run}_info.yaml"
                logger.info(f"{run_info_file=}")

                if download_runinfo:
                    if run_info_file.exists() and not force_download:
                        logger.info(f"File already exists, will not re-download: {run_info_file} ")
                    else:
                        if force_download:
                            logger.info("Force download enabled -- re-download all runinfo files...")
                        download_script = stat_xsede_2021_dir / "scripts/js_stat_xsede_steer/download_from_OSN.py"
                        cmd = f"python3 {download_script!s} -s {facility}/{run}/ -d {run_info_download_location} -f {run}_info.yaml"

                        # source = f'{facility}/{run}/{run}_info.yaml'
                        source = pathlib.Path(facility) / run / f"{run}_info.yaml"
                        destination = pathlib.Path(run_info_download_location) / f"{facility}/{run}/{run}_info.yaml"
                        # destination = os.path.join(run_info_download_location, f'{facility}/{run}/{run}_info.yaml')
                        destination.parent.mkdir(parents=True, exist_ok=True)

                        # append a tuple with .source and .target to the list
                        # runinfo_pairs.append as FilePair
                        runinfo_pairs.append(file_management.FilePair(source, destination))

            failed = []
            if runinfo_pairs:
                failed = file_management.download_from_OSN_pairs(runinfo_pairs, download_threads)

            if failed:
                logger.warning(f"Failed to download run_info.yaml for the following runs: {failed}")

            # need to loop over the runs again after processing is finished
            logger.info("Finished downloading run_info.yaml for all runs, building dictionaries ...")
            for i, run in enumerate(runs[facility].copy()):
                # should exist now after previous download
                run_info_file = run_info_download_location / f"{facility}/{run}/{run}_info.yaml"
                # Add the run_info block to the run_dictionary
                if run_info_file.exists():
                    with run_info_file.open() as f:
                        run_info = yaml.safe_load(f)
                        run_dictionary[facility][run]["calculation_type"] = run_info["calculation_type"]
                        run_dictionary[facility][run]["sqrt_s"] = run_info["sqrt_s"]
                        run_dictionary[facility][run]["centrality"] = run_info["centrality"]
                        if run_info["calculation_type"] == "jet_energy_loss":
                            if run_info["sqrt_s"] in [200]:
                                run_dictionary[facility][run]["system"] = "AuAu"
                            elif run_info["sqrt_s"] in [2760, 5020]:
                                run_dictionary[facility][run]["system"] = "PbPb"
                            run_dictionary[facility][run]["parametrization"] = run_info["parametrization"]

                            # Several options to skip runs we do not care about
                            if skip_run_binomial:
                                if run_info["parametrization"]["type"] == "binomial":
                                    # print(f'Skipping run {run} with binomial parametrization...')
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_parametrization_type) > 0:
                                if run_info["parametrization"]["type"] not in debug_parametrization_type:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_sqrts) > 0:
                                if run_info["sqrt_s"] not in debug_sqrts:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_centrality) > 0:
                                if run_dictionary[facility][run]["centrality"] not in debug_centrality:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_design_points) > 0:
                                if (
                                    run_dictionary[facility][run]["parametrization"]["design_point_index"]
                                    not in debug_design_points
                                ):
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_calculation_type) > 0:
                                if run_dictionary[facility][run]["calculation_type"] not in debug_calculation_type:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_system) > 0:
                                if run_dictionary[facility][run]["system"] not in debug_system:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                        else:
                            run_dictionary[facility][run]["system"] = "pp"
                            if len(debug_system) > 0:
                                if run_dictionary[facility][run]["system"] not in debug_system:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                            if len(debug_sqrts) > 0:
                                if run_dictionary[facility][run]["sqrt_s"] not in debug_sqrts:
                                    skipped_runs[facility].append(run)
                                    runs[facility].remove(run)
                                    continue
                else:
                    logger.warning(f"{run}_info.yaml not found!")
                    runs[facility].remove(run)
                    missing_runinfo[facility].append(run)

        # Print what we found
        for facility in facilities:
            msg = f"""{facility}:

  We found the following runs (N={len(runs[facility])}):

    {list(dict(run_dictionary[facility]).keys())}

"""
            logger.info(msg)
            msg_warning = f"""We did NOT find run_info for the following runs:")

    {missing_runinfo[facility]}

"""
            logger.warning(msg_warning)
            if len(skipped_runs[facility]) > 0:
                msg = f"""We skipped the following runs:
    {skipped_runs[facility]}

"""
                logger.warning(msg)

    # -----------------------------------------------------------------
    # Download histograms for each run
    # -----------------------------------------------------------------
    if download_histograms:
        logger.info("Downloading all histograms...\n")

        histogram_download_location = local_base_output_dir / "histograms_per_run"
        for facility in facilities:
            for run in runs[facility]:
                # If directory exists locally, check number of histograms already downloaded and number on OSN
                histogram_dir = histogram_download_location / f"{facility}/{run}/histograms"
                if histogram_dir.exists():
                    n_histograms_local = len([h for h in histogram_dir.iterdir() if h.suffix == ".root"])

                    script = stat_xsede_2021_dir / "scripts/js_stat_xsede_steer/count_files_on_OSN.py"
                    cmd = f"python3 {script!s} -s {facility}/{run}/version{analysis_version}/ -f histograms"
                    if analysis_version == -1:
                        cmd = f"python3 {script!s} -s {facility}/{run}/ -f histograms -x version"
                    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                    output = proc.stdout.read()
                    n_histograms_expected = int(output.split()[-1])

                # Download all histograms we are missing (or force download, if requested)
                if histogram_dir.exists() and n_histograms_expected == n_histograms_local and not force_download:
                    logger.info(
                        f"Histogram dir ({histogram_dir}) already exists and n_histograms_expected ({n_histograms_expected}) == n_histograms_local ({n_histograms_local}), will not re-download\n"
                    )
                else:
                    if force_download:
                        logger.info("Force download enabled -- re-download all histograms...")
                    if histogram_dir.exists() and n_histograms_expected != n_histograms_local:
                        logger.info(
                            f"Histogram dir already exists, but n_histograms_expected ({n_histograms_expected}) does not equal n_histograms_local ({n_histograms_local}), so we will redownload."
                        )

                    download_script = stat_xsede_2021_dir / "scripts/js_stat_xsede_steer/download_from_OSN.py"
                    cmd = f"python3 {download_script!s} -s {facility}/{run}/version{analysis_version}/ -d {histogram_download_location} -f histograms"
                    if analysis_version == -1:
                        cmd = f"python3 {download_script!s} -s {facility}/{run}/ -d {histogram_download_location} -f histograms -x version"
                    subprocess.run(cmd, check=True, shell=True)

        logger.info("Done!")
    if download_final_state_hadrons:
        # run the download server in the background as a subprocess
        from multiprocessing import Process

        download_server_process = Process(target=download_server, args=(facilities, runs))
        download_server_process.start()
        logger.info("Started download server in background...\n")

    if analysis_final_state_hadrons:
        logger.info("Downloading all final state hadrons...\n")

        # Some things for book keeping
        import shutil

        logger.info("Creating files for versioning book keeping...")
        config_path = jetscape_analysis_dir / "config"
        # search for STAT config files in config_path
        config_files = config_path.glob("STAT_*")
        # Setup
        reananalysis_output_location_base.mkdir(exist_ok=True, parents=True)
        for config_file in config_files:
            logger.info(f"Copying {config_file} to {reananalysis_output_location_base}...")
            shutil.copy(config_file, reananalysis_output_location_base)
        git_log_file = reananalysis_output_location_base / "git_log_jetscape-analysis.txt"
        subprocess.run(
            ["git", "log", "--oneline", "-n", "100"],
            check=False,
            cwd=jetscape_analysis_dir,
            stdout=git_log_file.open("w"),
        )

        # things for visualization
        total_runs = sum([len(runs[facility]) for facility in facilities])
        run_counter = 0
        for facility in facilities:
            for run in runs[facility]:
                # use for debugging only. Continue to take the same run for testing
                if do_debug_same_run:
                    run = runs[facility][0]
                run_counter += 1
                run_number = int(run[3:])

                logger.info(f"Processing run {run_number} ({run_counter}/{total_runs}) ...")
                # If directory exists locally, check number of histograms already downloaded and number on OSN
                final_state_hadron_dir = final_state_hadrons_download_location / f"{facility}/{run}"
                reananalysis_output_location = reananalysis_output_location_base / f"{facility}/{run}"

                # before we download any analysis output, check if we already have analysis output that we want. If it is already there then skip this run
                if analysis_final_state_hadrons and not force_reanalysis and not do_debug_same_run:
                    histpath = reananalysis_output_location / "histograms"
                    # before starting analysis check if output directory already exists
                    if histpath.exist():
                        # check number of output files
                        n_histoutput = len([h for h in histpath.iterdir() if h.suffix == ".root"])
                        n_parquet_local = len([h for h in final_state_hadron_dir.iterdir() if h.suffix == ".parquet"])

                        if n_histoutput == n_parquet_local:
                            logger.info(
                                f"Analysis already done for run {run_number}, I found {n_histoutput} histograms, will not re-analyze ..."
                            )
                            continue
                # check if file final_state_hadron_dir/DONE exist and the downloader finished buffering this run, if it does not exist, then wait until it exists
                while 1:
                    if (final_state_hadron_dir / "DONE").exists():
                        break
                    logger.info("Waiting for download of run to finish, re-checking in 60s...")
                    time.sleep(60)

                # only submit if the final_state_hadron_dir actually contains any files. For some weird reason sometimes files are not available for a run
                if (
                    len(list(final_state_hadron_dir.iterdir())) > 1
                ):  # bigger than 1 because of DONE file that might be there
                    arg_list = [
                        "python3",
                        f"{stat_xsede_2021_dir!s}/scripts/js_stat_xsede_steer/submit.py",
                        "analysis",
                        f"--run_number={run_number}",
                        f"--run_info_dir={run_info_download_location}",
                        f"--final_state_hadrons_dir={final_state_hadrons_download_location}",
                        f"--output_dir={reananalysis_output_location}",
                        f"--analysis_facility_name={local_analysis_facility}",
                        f"--download_facility_name={facility}",
                        f"--container_path={analysis_container_path}",
                        f"--n_tasks={n_tasks}",
                        f"--max_n_slurm_jobs={max_n_slurm_jobs}",
                        f"--jetscape_analysis_dir={jetscape_analysis_dir!s}",
                    ]
                    subprocess.run(arg_list, check=True, shell=False)
                else:
                    logger.info(f"No final state hadrons found for run {run_number}, skipping analysis...")
                    # append run to list skipped_runs.log in final_state_hadrons_download_location. Create the file if it does not exist
                    skipped_runs_file = final_state_hadrons_download_location / "skipped_runs.log"
                    with skipped_runs_file.open("a") as f:
                        f.write(f"{run_number}\n")
                if delete_local_final_state_hadrons_after_analysis:
                    # safely delete final_state_hadron_dir, make sure the path contains final_state_hadrons_per_run so we don't by accident delete "/" or "/home" ....

                    if final_state_hadron_dir.exists() and "final_state_hadrons_per_run" in str(final_state_hadron_dir):
                        # remove all files in the directory that contain final_state_hadrons and parquet in name
                        # note that we don't ask for ending in parquet since sometimes incomplete files have weird ending
                        logger.info(f"Deleting all parquet files in {final_state_hadron_dir} as requested...")
                        for f in final_state_hadron_dir.iterdir():
                            if "final_state_hadrons" in f.name and "parquet" in f.name:
                                f.unlink()
                            # remove DONE file
                            if "DONE" in f.name:
                                f.unlink()
                        # now we can delete the directory
                        final_state_hadron_dir.rmdir()

                # TODO upload analysis results / histos to OSN in the future

        logger.info("Done!")

    # -----------------------------------------------------------------
    # Merge for each run into a single histogram per run
    # -----------------------------------------------------------------
    if merge_histograms:
        for facility in facilities:
            run_counter = 0
            for run in runs[facility]:
                run_counter += 1
                logger.info(f"Processing run {run_counter} out of {len(runs[facility])} at facility {facility}...")
                outputdir = local_base_output_dir / f"histograms_per_run/{facility}/{run}"
                inputdir = outputdir / "histograms"

                if outputdir.exists():
                    ROOT_filenames = (reananalysis_output_location_base / f"{facility}/{run}/histograms").iterdir()
                    file_list = outputdir / "files_to_merge.txt"
                    with file_list.open("w") as f:
                        for filename in ROOT_filenames:
                            f.write(f"{filename!s}\n")

                    system = run_dictionary[facility][run]["system"]
                    sqrts = run_dictionary[facility][run]["sqrt_s"]
                    fname = f"histograms_{system}_{run}_{sqrts}.root"

                    cmd = f"hadd -j {n_cores} -f {outputdir / fname!s} @{file_list}"
                    subprocess.run(cmd, check=True, shell=True)
                    file_list.unlink()

    # -----------------------------------------------------------------
    # List histogram paths for selected design points.
    # Used to extract unmerged histograms for separate studies.
    # -----------------------------------------------------------------
    if list_paths_for_selected_design_points:
        # These options are only put here since they're quite niche.
        selected_parametrization = "exponential"
        selected_design_point_indices = list(range(40))
        selected_centrality = [0, 10]
        # Using a relative directory here is useful since we will want to tar these files up
        # while keeping a directory structure
        relative_dir = Path(local_base_output_dir)

        # Setup
        # NOTE: We'll store a list of paths per design index because we have multiple sqrt_s per design point
        design_point_index_to_path = defaultdict(list)

        for facility in facilities:
            for run in runs[facility]:
                filepath_base = local_base_output_dir / f"histograms_per_run/{facility}/{run}"
                if not list(filepath_base.rglob("*.root")):
                    continue

                sqrts = run_dictionary[facility][run]["sqrt_s"]
                system = run_dictionary[facility][run]["system"]
                if run_dictionary[facility][run]["calculation_type"] == "jet_energy_loss":
                    # AA case
                    centrality = run_dictionary[facility][run]["centrality"]
                    parametrization = run_dictionary[facility][run]["parametrization"]
                    parametrization_type = parametrization["type"]
                    design_point_index = parametrization["design_point_index"]

                    # Apply selection
                    if not (
                        centrality == selected_centrality
                        and design_point_index in selected_design_point_indices
                        and parametrization_type == selected_parametrization
                    ):
                        continue
                else:
                    # pp case
                    # We define -1 as the convention
                    design_point_index = -1
                    continue

                design_point_index_to_path[design_point_index].append(filepath_base)

        # Sort the output
        # NOTE: We don't sort by sqrt_s because we don't have that info. It's not so important in any case.
        design_point_index_to_path = dict(sorted(design_point_index_to_path.items()))
        output_files = []
        for paths in design_point_index_to_path.values():
            for path in paths:
                # We want two outputs:
                # 1. The merged histogram, if available.
                # NOTE: We need to search for the histogram before we convert it to the relative path!
                output_files.extend([p.relative_to(relative_dir) for p in path.glob("histograms_*.root")])
                # 2. The unmerged histograms directory.
                output_files.append(path.relative_to(relative_dir) / "histograms")

        # And print it for the user to utilize as desired
        msg = f"""We found the following paths for the selected design points:

{design_point_index_to_path}

  List of files:

"""
        msg += "    " + " ".join([str(s) for s in output_files])
        logger.info(msg)

    # -----------------------------------------------------------------
    # Aggregate histograms for runs with a common: (sqrts, system, parametrization_type, design_point_index)
    # We sum over: facilities, run numbers, centralities
    #
    # Write dictionary to design_point_info.pkl that stores relevant info for each design point
    #
    # Note that we store xsec and weight_sum separately for each centrality,
    # such that we can merge before running plotting script
    # -----------------------------------------------------------------
    if aggregate_histograms:
        logger.info("Aggregate histograms for runs with common (sqrts, system, parametrization, design_point_index)\n")

        # Create a dict that stores list of local paths for each aggregated histogram:
        #   design_point_dictionary[(sqrts, system, parametrization_type, design_point_index)] = ['path/to/run1', 'path/to/run2', ...]
        design_point_dictionary = {}
        for facility in facilities:
            for run in runs[facility]:
                filepath_base = local_base_output_dir / f"histograms_per_run/{facility}/{run}"
                if not list(filepath_base.rglob("*.root")):
                    continue

                sqrts = run_dictionary[facility][run]["sqrt_s"]
                system = run_dictionary[facility][run]["system"]
                if run_dictionary[facility][run]["calculation_type"] == "jet_energy_loss":
                    parametrization = run_dictionary[facility][run]["parametrization"]
                    parametrization_type = parametrization["type"]
                    design_point_index = parametrization["design_point_index"]
                else:
                    parametrization = None
                    parametrization_type = None
                    design_point_index = None

                design_point_tuple = (sqrts, system, parametrization_type, design_point_index)
                if design_point_tuple not in design_point_dictionary:
                    design_point_dictionary[design_point_tuple] = defaultdict(list)

                filepath = filepath_base / f"histograms_{system}_{run}_{sqrts}.root"
                design_point_dictionary[design_point_tuple]["files"].append(filepath)
                design_point_dictionary[design_point_tuple]["parametrization"] = parametrization

        # Merge each list of histograms together, and write into a new directory structure
        outputdir_base = local_base_output_dir / "histograms_aggregated"
        for design_point_tuple in design_point_dictionary:
            logger.info(f"{design_point_tuple=}\n")
            sqrts, system, parametrization_type, design_point_index = design_point_tuple

            if parametrization_type:
                fname = f"histograms_design_point_{design_point_index}.root"
                outputdir = outputdir_base / f"{sqrts}_{system}_{parametrization_type}"
            else:
                fname = "histograms.root"
                outputdir = outputdir_base / f"{sqrts}_{system}"
            outputdir.mkdir(exist_ok=True, parents=True)

            cmd = f"hadd -f {outputdir / fname!s}"
            for filepath in design_point_dictionary[design_point_tuple]["files"]:
                cmd += f" {filepath!s}"
            subprocess.run(cmd, check=True, shell=True)
            design_point_dictionary[design_point_tuple]["histogram_aggregated"] = outputdir / fname

        # Write design_point_dictionary to file
        outfile = outputdir_base / "design_point_info.pkl"
        with outfile.open("wb") as f:
            pickle.dump(design_point_dictionary, f)
    elif not (local_base_output_dir / "histograms_aggregated" / "design_point_info.pkl").exists():
        # If we don't have the design_point_info, we need to try to reproduce it as best as we can
        # We'll base it on the run_info file (assumes that we have histograms for each run, so this is less than ideal)
        # TODO(RJE): It would be best if this could be consolidated. However, this was useful for testing in 2026 March.
        logger.info("Attempting to create the design_point_info manually. This will not work if there's missing data!")

        # Create a dict that stores list of local paths for each aggregated histogram:
        #   design_point_dictionary[(sqrts, system, parametrization_type, design_point_index)] = ['path/to/run1', 'path/to/run2', ...]
        design_point_dictionary = {}
        for facility in facilities:
            for run in runs[facility]:
                # NOTE: This path may or may not exist (depending on whether we've done histograms_per_run aggregation),
                #       but it's the path that would be expected, so we include it anyway for consistency
                filepath_base = local_base_output_dir / f"histograms_per_run/{facility}/{run}"

                sqrts = run_dictionary[facility][run]["sqrt_s"]
                system = run_dictionary[facility][run]["system"]
                if run_dictionary[facility][run]["calculation_type"] == "jet_energy_loss":
                    parametrization = run_dictionary[facility][run]["parametrization"]
                    parametrization_type = parametrization["type"]
                    design_point_index = parametrization["design_point_index"]
                else:
                    parametrization = None
                    parametrization_type = None
                    design_point_index = None

                design_point_tuple = (sqrts, system, parametrization_type, design_point_index)
                if design_point_tuple not in design_point_dictionary.keys():
                    design_point_dictionary[design_point_tuple] = defaultdict(list)

                filepath = filepath_base / f"histograms_{system}_{run}_{sqrts}.root"
                design_point_dictionary[design_point_tuple]["files"].append(filepath)
                design_point_dictionary[design_point_tuple]["parametrization"] = parametrization

        # Determine the name of the aggregated histogram
        outputdir_base = local_base_output_dir / "histograms_aggregated"
        for design_point_tuple in design_point_dictionary.keys():
            logger.info("{design_point_tuple=}\n")
            sqrts, system, parametrization_type, design_point_index = design_point_tuple

            if parametrization_type:
                fname = f"histograms_design_point_{design_point_index}.root"
                outputdir = outputdir_base / f"{sqrts}_{system}_{parametrization_type}"
            else:
                fname = "histograms.root"
                outputdir = outputdir_base / f"{sqrts}_{system}"

            design_point_dictionary[design_point_tuple]["histogram_aggregated"] = outputdir / fname

        # Write design_point_dictionary to file
        outfile = outputdir_base / "design_point_info.pkl"
        with outfile.open("wb") as f:
            pickle.dump(design_point_dictionary, f)

        logger.info(f"Wrote design_point_info.pkl to {outfile}")

    # -----------------------------------------------------------------
    # Plot histograms and save to ROOT files
    # -----------------------------------------------------------------
    if plot_and_save_histograms:
        # Get dictionary containing info for each design point: (sqrts, system, parametrization_type, design_point_index)
        outputdir_base = local_base_output_dir / "histograms_aggregated"
        outfile = outputdir_base / "design_point_info.pkl"
        with outfile.open("rb") as f:
            design_point_dictionary = pickle.load(f)

        # First plot pp
        for design_point_tuple in design_point_dictionary.keys():
            sqrts, system, parametrization_type, design_point_index = design_point_tuple
            if system == "pp":
                outputdir = local_base_output_dir / f"plot/{sqrts}_{system}"
                inputfile = outputdir_base / f"{sqrts}_{system}/histograms.root"
                cmd = f"python3 {jetscape_analysis_dir!s}/plot/plot_results_STAT.py"
                cmd += f" -c {jetscape_analysis_dir!s}/config/STAT_{sqrts}.yaml"
                cmd += f" -i {inputfile!s}"
                cmd += f" -o {outputdir!s}"
                cmd += f" -m {model_name}"
                subprocess.run(cmd, check=True, shell=True)

        # Then plot AA, using appropriate pp reference, and construct AA/pp ratios
        process_list = []
        for design_point_tuple in design_point_dictionary.keys():
            sqrts, system, parametrization_type, design_point_index = design_point_tuple
            if system in ["AuAu", "PbPb"]:
                outputdir = local_base_output_dir / f"plot/{sqrts}_{system}_{parametrization_type}/{design_point_index}"
                # TODO(RJE): Should reconcile the leading 0 convention...
                #            I tried to include a leading 0 for single digit run numbers in the aggregation, but
                #            we generally don't do that. I can change it here, but it needs to be propagated elsewhere.
                inputfile = (
                    outputdir_base
                    / f"{sqrts}_{system}_{parametrization_type}/histograms_design_point_{design_point_index}.root"
                )
                pp_reference_filename = local_base_output_dir / f"plot/{sqrts}_pp/final_results.root"
                if pp_reference_filename.exists():
                    cmd = f"python3 {jetscape_analysis_dir!s}/plot/plot_results_STAT.py"
                    cmd += f" -c {jetscape_analysis_dir!s}/config/STAT_{sqrts}.yaml"
                    cmd += f" -i {inputfile!s}"
                    cmd += f" -r {pp_reference_filename!s}"
                    cmd += f" -o {outputdir!s}"
                    cmd += f" -m {model_name}"

                    # Execute in parallel
                    # Simple & quick implementation: once max_processes have been launched, wait for them to finish before continuing
                    # NOTE: Needs to be Popen so it's not blocking
                    process = subprocess.Popen(cmd, shell=True)
                    process_list.append(process)
                    if len(process_list) > n_cores - 1:
                        for subproc in process_list:
                            subproc.wait()
                        process_list = []

    # -----------------------------------------------------------------
    # Convert histograms to data tables and save
    # -----------------------------------------------------------------
    if write_tables:
        # Construct table of model predictions for all design points, as input to Bayesian analysis
        # We can easily adapt this to the format v1.0 specified here, although we may want to update a bit: https://www.evernote.com/l/ACWFCWrEcPxHPJ3_P0zUT74nuasCoL_DBmY
        plot_dir = local_base_output_dir / "plot"
        table_base_dir = local_base_output_dir / "tables"
        prediction_table_dir = table_base_dir / "Prediction"
        data_table_dir = table_base_dir / "Data"
        design_table_dir = table_base_dir / "Design"

        prediction_table_dir.mkdir(exist_ok=True, parents=True)
        data_table_dir.mkdir(exist_ok=True, parents=True)
        design_table_dir.mkdir(exist_ok=True, parents=True)

        # We will write out design point files as well
        design_point_dir = local_base_output_dir / "histograms_aggregated"
        outfile = design_point_dir / "design_point_info.pkl"
        with outfile.open("rb") as f:
            design_point_dictionary = pickle.load(f)
        design_df = {}

        # Loop through each directory corresponding to a given (sqrts, parameterization)
        for label_dir in plot_dir.iterdir():
            label = label_dir.name
            if "AuAu" in label or "PbPb" in label:
                sqrts, system, parameterization = label.split("_")

                output_dict = defaultdict()
                output_dict["values"] = {}
                output_dict["errors"] = {}
                output_dict["bin_edges"] = {}
                output_dict["observable_label"] = {}

                # Loop through design points and observables, and construct a dataframe of predictions for each observable
                #   columns=[design1, design2, ...]
                #   rows=[bin1, bin2, ...]
                logger.info(f"Constructing prediction dataframes for {label}...")
                for design_point_index in label_dir.iterdir():
                    # if design_point_index != '0':
                    #    continue
                    if not design_point_index.is_file() and design_point_index.name != "Data":
                        # print(f'  design_point_index: {design_point_index}')

                        final_result_h5 = design_point_index / "final_results.h5"
                        if not final_result_h5.exists():
                            continue

                        with h5py.File(final_result_h5, "r") as hf:
                            for key in list(hf.keys()):
                                # Use a separate dataframe for values, errors, bin_edges
                                if "values" in key:
                                    type = "values"
                                elif "errors" in key:
                                    type = "errors"
                                elif "bin_edges" in key:
                                    type = "bin_edges"
                                else:
                                    sys.exit(f"Unexpected key: {key}")

                                # Get observable label for bookkeeping
                                observable_labels = [
                                    s.name.replace("h_", "", 1).replace(".pdf", "")
                                    for s in design_point_index.iterdir()
                                    if s.suffix == ".pdf"
                                ]
                                output_dict["observable_label"][key] = None
                                for s in observable_labels:
                                    if s in key:
                                        output_dict["observable_label"][key] = s

                                # Put design point info into dataframe, with design point index as index of dataframe
                                if key not in output_dict[type]:
                                    output_dict[type][key] = pd.DataFrame()

                                output_dict[type][key][f"design_point{design_point_index.name}"] = hf[key][:]

                                design_point_key = (int(sqrts), system, parameterization, int(design_point_index.name))
                                parameterization_values = pd.DataFrame(
                                    data=[
                                        design_point_dictionary[design_point_key]["parametrization"][
                                            "parametrization_values"
                                        ]
                                    ],
                                    index=[int(design_point_index.name)],
                                )
                                if parameterization not in design_df:
                                    design_df[parameterization] = parameterization_values
                                elif int(design_point_index.name) not in design_df[parameterization].index:
                                    design_df[parameterization] = pd.concat(
                                        [design_df[parameterization], parameterization_values]
                                    )
                logger.info(f"Done constructing prediction dataframes for {label}.\n")

                # Write Prediction and Data dataframes to txt
                logger.info(f"Writing prediction tables for {label}...")
                for type in output_dict.keys():
                    if type in ["observable_label", "bin_edges"]:
                        continue

                    for key, df in output_dict[type].items():
                        key_items = key.split("_")
                        # if 'values' in key:
                        #    print()
                        #    print(key_items)

                        # Parse observable-specific names -- there are a few different cases depending on the observable class
                        # We uniformize the structure as: f'{parameterization}__{sqrts}__{system}__{observable_category}__{observable}__{subobservable}__{centrality[0]}-{centrality[1]}'
                        # For example: binomial__5020__PbPb__inclusive_chjet__pt_alice__R0.4__0-10
                        # This will allow us to easily parse the observables in a uniform way, and also access info in the STAT observable config files
                        # Note that the experiment can always be accessed as observable.split('_')[-1]
                        # The subobservable can include: jet radius, grooming condition, pt bin index

                        # Hadron observables -- after hole subtraction
                        if "hadron" in key and "unsubtracted" not in key:
                            observable_category = key_items[2]
                            observable = f"{key_items[3]}_{key_items[4]}_{key_items[5]}"
                            subobservable = ""
                            centrality = ["".join(filter(str.isdigit, s)) for s in key_items[6].split(",")]

                        # Jet observables -- with negative_recombiner
                        # There are several different subobservable patterns that we need to parse
                        elif "negative_recombiner" in key:
                            if "zcut" in key:
                                observable_category = f"{key_items[4]}_{key_items[5]}"
                                observable = f"{key_items[6]}_{key_items[7]}"
                                subobservable = f"{key_items[8]}_{key_items[9]}_{key_items[10]}"
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[11].split(",")]
                            elif "charge" in key:
                                observable_category = f"{key_items[4]}_{key_items[5]}"
                                observable = f"{key_items[6]}_{key_items[7]}"
                                subobservable = f"{key_items[8]}_{key_items[9]}"
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[10].split(",")]
                            elif "dijet_trigger_jet" in key:
                                observable_category = key_items[4]
                                observable = f"{key_items[5]}_{key_items[6]}"
                                subobservable = f"{key_items[7]}_{key_items[9]}"
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[8].split(",")]
                            elif "_y_" in key:
                                observable_category = f"{key_items[4]}_{key_items[5]}"
                                observable = f"{key_items[6]}_{key_items[7]}_{key_items[8]}"
                                subobservable = f"{key_items[9]}_{key_items[11]}"
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[10].split(",")]
                            elif key_items[-2] in [f"pt{i}" for i in range(10)]:
                                observable_category = f"{key_items[4]}_{key_items[5]}"
                                observable = f"{key_items[6]}_{key_items[7]}"
                                subobservable = f"{key_items[8]}_{key_items[10]}"
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[9].split(",")]
                            else:
                                observable_category = f"{key_items[4]}_{key_items[5]}"
                                observable = f"{key_items[6]}_{key_items[7]}"
                                subobservable = key_items[8]
                                centrality = ["".join(filter(str.isdigit, s)) for s in key_items[9].split(",")]

                        # Skip other observables (hadrons without subtraction, jets with other subtraction schemes)
                        else:
                            if (
                                "shower_recoil" not in key
                                and "constituent_subtraction" not in key
                                and "unsubtracted" not in key
                            ):
                                logger.info(f"Unexpected key: {key}")
                            continue

                        observable_name_data = f"{sqrts}__{system}__{observable_category}__{observable}__{subobservable}__{centrality[0]}-{centrality[1]}"
                        observable_name_prediction = f"{parameterization}__{observable_name_data}"

                        if "values" in key:
                            logger.info(f"  {observable_name_prediction}")

                        # Sort columns
                        df_prediction = output_dict[type][key]
                        df_prediction = df_prediction.reindex(
                            sorted(df_prediction.columns, key=lambda x: float(x[12:])), axis=1
                        )
                        columns = list(df_prediction.columns)

                        # Get experimental data
                        observable_label = output_dict["observable_label"][key]
                        data_dir = label_dir / "Data"
                        filename = f"Data_{observable_label}.dat"
                        data_file = data_dir / filename
                        if data_file.exists():
                            data = np.loadtxt(data_file, ndmin=2)

                        if df_prediction.to_numpy().shape[0] != data.shape[0]:
                            logger.info(
                                f"Mismatch of number of bins: prediction ({df_prediction.to_numpy().shape[0]}) vs. data ({data.shape[0]})"
                            )

                        # Remove rows with leading zeros (corresponding to bins below the min_pt cut)
                        n_zero_rows = 0
                        for row in df_prediction.to_numpy():
                            if np.all(row == 0):
                                n_zero_rows += 1
                            else:
                                break
                        df_prediction = df_prediction.iloc[n_zero_rows:, :]
                        data = data[n_zero_rows:, :]

                        if df_prediction.to_numpy().shape[0] != data.shape[0]:
                            logger.info(
                                f"Mismatch of number of bins after removing zeros: prediction ({df_prediction.to_numpy().shape[0]}) vs. data ({data.shape[0]})\n"
                            )

                        # Write Prediction.dat and Data.dat files
                        filename = prediction_table_dir / f"Prediction__{observable_name_prediction}__{type}.dat"
                        design_point_file = f"Design_{parameterization}.dat"
                        header = (
                            f"Version 2.0\nData Data_{observable_name_prediction}.dat\nDesign {design_point_file}\n"
                        )
                        header += " ".join(columns)
                        np.savetxt(filename, df_prediction.values, header=header)

                        filename = data_table_dir / f"Data__{observable_name_data}.dat"
                        header = "Version 1.1\n"
                        header += "Label xmin xmax y y_err"
                        np.savetxt(filename, data, header=header)

                logger.info(f"Done writing prediction tables for {label}.\n")

        # Write out the Design.dat files
        logger.info("Writing design point tables...")
        for parameterization in design_df:
            # Sort according to index
            df = design_df[parameterization].sort_index()

            # Rename columns
            if model_name == "hybrid":
                # No need to rename as of 2026 March

                # Reorder columns to match previous convention
                ordered_columns = ["L_res", "kappa_sc"]
            else:
                df.rename(
                    columns={"t_start": "Tau0", "alpha_s": "AlphaS", "q_switch": "Q0", "A": "C1", "B": "C2"},
                    inplace=True,
                )
                if parameterization == "binomial":
                    df.rename(columns={"C": "A", "D": "B"}, inplace=True)
                elif parameterization == "exponential":
                    df.rename(columns={"C": "C3"}, inplace=True)

                # Reorder columns to match previous convention
                if parameterization == "binomial":
                    ordered_columns = ["AlphaS", "Q0", "C1", "C2", "Tau0", "A", "B"]
                elif parameterization == "exponential":
                    ordered_columns = ["AlphaS", "Q0", "C1", "C2", "Tau0", "C3"]

            df = df[ordered_columns]

            # Write
            filename = design_table_dir / f"Design__{parameterization}.dat"
            header = "Version 1.0\n"
            header += f"- Design points for {parameterization} PDF\n"
            parameters = " ".join(df.keys())
            header += f"Parameter {parameters}\n"
            header += "- Parameter AlphaS: Linear [0.1, 0.5]\n"
            header += "- Parameter Q0: Linear [1, 10]\n"
            header += "- Parameter C1: Log [0.006737946999085467, 10]\n"
            header += "- Parameter C2: Log [0.006737946999085467, 10]\n"
            header += "- Parameter Tau0: Linear [0.0, 1.5]\n"
            if parameterization == "binomial":
                header += "- Parameter A: Linear [-10, 100]\n"
                header += "- Parameter B: Linear [-10, 100]\n"
            elif parameterization == "exponential":
                header += "- Parameter C3: Log [0.049787068367863944, 100]\n"
            indices = " ".join([str(i) for i in df.index])
            header += f"Design point indices (row index): {indices}"
            np.savetxt(filename, df.values, header=header)

        logger.info("Done!")

    # -----------------------------------------------------------------
    # Plot some QA over all aggregated runs
    # -----------------------------------------------------------------
    if plot_global_QA:
        histograms_aggregated_dir = local_base_output_dir / "histograms_aggregated"
        plot_dir = local_base_output_dir / "plot"
        global_qa_dir = local_base_output_dir / "global_qa"
        global_qa_dir.mkdir(exist_ok=True, parents=True)

        # ----------------------
        # Plot n_events
        # Make a 2d plot from a 2d numpy array: n_generated/n_target for (sqrts_parameterization_centrality, design_point_index)
        if model_name == "hybrid":
            n_design_points = 40
            n_design_point_max = {"Lres-E-loss": 40}
            n_systems = 2
            shape = (n_systems, n_design_points)
            n_events = np.zeros(shape)
            n_target = {"5020": 2000000}
        else:
            n_design_points = 230
            n_design_point_max = {"exponential": 230, "binomial": 180}
            n_systems = 12
            shape = (n_systems, n_design_points)
            n_events = np.zeros(shape)
            n_target = {"200": 500000, "2760": 442000, "5020": 1700000}

        # Keep track of which design points need to be re-run
        rerun_threshold = 0.75
        rerun_dict = defaultdict(list)
        missing_dict = defaultdict(list)

        # Loop through each directory corresponding to a given (sqrts, parameterization)
        system_index = 0
        system_labels = []
        for sqrts_parameterization_label in plot_dir.iterdir():
            if "AuAu" in sqrts_parameterization_label.name or "PbPb" in sqrts_parameterization_label.name:
                sqrts, system, parameterization = sqrts_parameterization_label.name.split("_")
                qa_plot_dir = sqrts_parameterization_label

                for design_point_index in range(n_design_points):
                    if design_point_index >= n_design_point_max[parameterization]:
                        continue

                    # TODO(RJE): These aren't really the right ranges for the hybrid model production
                    if not (qa_plot_dir / str(design_point_index)).exists():
                        missing_dict[f"{sqrts_parameterization_label.name}_0-10"].append(int(design_point_index))
                        missing_dict[f"{sqrts_parameterization_label.name}_10-50"].append(int(design_point_index))
                        continue

                    fname = f"{sqrts_parameterization_label.name}/histograms_design_point_{design_point_index}.root"
                    f = ROOT.TFile(str(histograms_aggregated_dir / fname), "read")
                    # TODO(RJE): I'm not sure this value is set correctly in the analysis...
                    h = f.Get("h_centrality_range_generated")
                    ratio_0_10 = (
                        h.Integral(h.GetXaxis().FindBin(0 + 0.5), h.GetXaxis().FindBin(10 - 0.5)) / n_target[sqrts]
                    )
                    ratio_10_50 = (
                        h.Integral(h.GetXaxis().FindBin(10 + 0.5), h.GetXaxis().FindBin(50 - 0.5)) / n_target[sqrts]
                    )

                    n_events[system_index, int(design_point_index)] = ratio_0_10
                    n_events[system_index + 1, int(design_point_index)] = ratio_10_50

                    if 0.01 < ratio_0_10 < rerun_threshold:
                        rerun_dict[f"{sqrts_parameterization_label.name}_0-10"].append(int(design_point_index))
                    if 0.01 < ratio_10_50 < rerun_threshold:
                        rerun_dict[f"{sqrts_parameterization_label.name}_10-50"].append(int(design_point_index))
                    if np.isclose(ratio_0_10, 0.0):
                        missing_dict[f"{sqrts_parameterization_label.name}_0-10"].append(int(design_point_index))
                    if np.isclose(ratio_10_50, 0.0):
                        missing_dict[f"{sqrts_parameterization_label.name}_10-50"].append(int(design_point_index))

                system_labels.append(f"{sqrts_parameterization_label.name}_0-10")
                system_labels.append(f"{sqrts_parameterization_label.name}_10-50")
                system_index += 2

        # Order the systems
        if model_name == "hybrid":
            ordered_indices = [0, 1]
            n_events[:] = n_events[ordered_indices, :]
            system_labels = [system_labels[i] for i in ordered_indices]
        else:
            ordered_indices = [3, 2, 11, 10, 9, 8, 5, 4, 7, 6, 1, 0]
            n_events[:] = n_events[ordered_indices, :]
            system_labels = [system_labels[i] for i in ordered_indices]

        # Plot
        fig, ax = plt.subplots()
        fig.suptitle(r"Number of events: $N_{gen} / N_{target}$", fontsize=16)
        c = ax.imshow(n_events, cmap="jet", aspect="auto", vmin=0.0, vmax=2.0, interpolation="nearest")
        fig.colorbar(c)
        ax.set_xlabel("Design point index", size=12)
        system_ticks = np.linspace(0, n_systems - 1, n_systems)
        plt.yticks(system_ticks, system_labels, size=6)
        outfilename = global_qa_dir / "n_events.pdf"
        plt.tight_layout()
        plt.savefig(outfilename)
        logger.info("Plotted n_events.")

        # Print which design points needs to be rerun
        msg = f"""

The following design points had fewer than {rerun_threshold}x the target statistics
"""
        for key, val in rerun_dict.items():
            msg += f"  {key}: {sorted(val)}"
        msg += "\nThe following design points have not yet been uploaded\n"
        for key, val in missing_dict.items():
            msg += f"  {key}: {sorted(val)}"
        msg += "\n"
        logger.info(msg)

        # ----------------------
        # Plot statistical uncertainty for each bin of each observable (for a given design point)
        table_dir = local_base_output_dir / "tables"
        prediction_dir = table_dir / "Prediction"
        data_dir = table_dir / "Data"

        if model_name == "hybrid":
            parameterizations = ["Lres-E-loss"]
        else:
            parameterizations = ["exponential", "binomial"]
        n_bins = 31

        # Construct 3d array: relative statistical uncertainty for (design_point_index, observable, bin)
        for parameterization in parameterizations:
            n_observables = len(
                [
                    x
                    for x in prediction_dir.iterdir()
                    if "Prediction" in x.name and "values" in x.name and parameterization in x.name
                ]
            )
            shape = (n_bins, n_design_point_max[parameterization], n_observables)
            relative_uncertainty_prediction = np.zeros(shape)
            relative_uncertainty_ratio_to_data = np.zeros(shape)

            observable_labels = []
            i_observable = 0
            for file_prediction in prediction_dir.iterdir():
                # file_prediction_keys = file_prediction.split('__')
                if (
                    "Prediction" in file_prediction.name
                    and "values" in file_prediction.name
                    and parameterization in file_prediction.name
                ):
                    observable = file_prediction.name[12:-12].replace(f"{parameterization}__", "")
                    file_prediction_errors = file_prediction.parent / file_prediction.name.replace("values", "errors")

                    # Get predictions and compute relative uncertainty -- zero pad to fixed size
                    prediction_values = np.loadtxt(file_prediction, ndmin=2)
                    prediction_errors = np.loadtxt(file_prediction_errors, ndmin=2)
                    if 0 in prediction_values:
                        logger.warning(
                            f"{file_prediction} has value=0 at design points {np.where(prediction_values == 0)[1]}"
                        )

                    relative_uncertainty_prediction_unpadded = np.divide(prediction_errors, prediction_values)
                    if relative_uncertainty_prediction_unpadded.shape[0] > n_bins:
                        msg = f"Set n_bins to be {relative_uncertainty_prediction_unpadded.shape[0]} or larger (due to {observable})"
                        raise ValueError(msg)

                    observable_shape = relative_uncertainty_prediction_unpadded.shape
                    n_pads_x = shape[0] - observable_shape[0]
                    n_pads_y = shape[1] - observable_shape[1]
                    relative_uncertainty_prediction_padded = np.pad(
                        relative_uncertainty_prediction_unpadded, ((0, n_pads_x), (0, n_pads_y))
                    )
                    relative_uncertainty_prediction[:, :, i_observable] = relative_uncertainty_prediction_padded

                    # Get data and compute relative uncertainty -- zero pad to fixed size
                    file_data = f"Data__{observable}.dat"
                    data = np.loadtxt(data_dir / file_data, ndmin=2)
                    data_values = data[:, 2]
                    data_errors = data[:, 3]
                    if 0 in data_values:
                        logger.warning(
                            f"{file_data} has value=0 at design points {np.where(prediction_values == 0)[1]}"
                        )

                    relative_uncertainty_data_unpadded = np.divide(data_errors, data_values)

                    # Compute ratio of prediction uncertainty to data uncertainty -- zero pad to fixed size
                    if prediction_values.shape[0] != data_values.shape[0]:
                        sys.exit(
                            f"({observable_shape}) has different shape than Data ({relative_uncertainty_data_unpadded.shape})"
                        )

                    relative_uncertainty_ratio_to_data_unpadded = np.divide(
                        relative_uncertainty_prediction_unpadded, relative_uncertainty_data_unpadded[:, None]
                    )
                    relative_uncertainty_ratio_to_data_padded = np.pad(
                        relative_uncertainty_ratio_to_data_unpadded, ((0, n_pads_x), (0, n_pads_y))
                    )
                    relative_uncertainty_ratio_to_data[:, :, i_observable] = relative_uncertainty_ratio_to_data_padded

                    observable_labels.append(observable)
                    i_observable += 1

            # TODO: Do something about bins that have value=0 and cause division problem, leading to empty entry

            # Order the observables by sqrts, and then alphabetically (i.e. by observable and centrality)
            ordered_labels = np.sort(observable_labels).tolist()
            ordered_indices = np.argsort(observable_labels).tolist()
            ordered_indices_200 = [ordered_indices[i] for i in range(n_observables) if "200" in ordered_labels[i]]
            ordered_indices_2700 = [ordered_indices[i] for i in range(n_observables) if "2760" in ordered_labels[i]]
            ordered_indices_5020 = [ordered_indices[i] for i in range(n_observables) if "5020" in ordered_labels[i]]
            ordered_indices = ordered_indices_5020 + ordered_indices_2700 + ordered_indices_200

            relative_uncertainty_prediction[:] = relative_uncertainty_prediction[:, :, ordered_indices]
            relative_uncertainty_ratio_to_data[:] = relative_uncertainty_ratio_to_data[:, :, ordered_indices]
            observable_labels_ordered = [observable_labels[i] for i in ordered_indices]
            logger.info(f"\n(n_bins, n_design_points, n_observables_total) = {relative_uncertainty_prediction.shape}")

            # Group observables and make a plot for each group
            groups = ["hadron__pt_", "jet__pt_", "Dpt", "Dz", ""]
            group_indices_total = []
            for group in groups:
                if group:  # Plot all observables containing a matching string
                    group_indices = [i for i, label in enumerate(observable_labels_ordered) if group in label]
                    group_indices_total += group_indices
                else:  # Plot everything that remains
                    group_indices = [i for i, _ in enumerate(observable_labels_ordered) if i not in group_indices_total]
                    group = "other"  # noqa: PLW2901
                n_observables_group = len(group_indices)
                logger.info(f"n_observables ({group}) = {n_observables_group}")

                group_mask = np.array([i in group_indices for i, _ in enumerate(observable_labels_ordered)])
                observable_labels_ordered_group = [observable_labels_ordered[i] for i in group_indices]

                relative_uncertainty_prediction_group = relative_uncertainty_prediction[:, :, group_mask]
                relative_uncertainty_ratio_to_data_group = relative_uncertainty_ratio_to_data[:, :, group_mask]

                # Plot relative uncertainty of prediction
                fig = plt.figure(figsize=[12, 15])
                ax = plt.axes()
                fig.suptitle("% statistical uncertainty on prediction -- mean", fontsize=24)
                matrix = np.transpose(np.mean(relative_uncertainty_prediction_group, axis=1))
                matrix_masked = np.ma.masked_where((matrix < 1e-8), matrix)
                c = ax.imshow(matrix_masked, cmap="jet", aspect="auto", vmin=0.0, vmax=0.2, interpolation="nearest")
                fig.colorbar(c)
                ax.set_xlabel("Observable bin", size=16)
                bin_ticks = range(n_bins)
                plt.xticks(bin_ticks, bin_ticks, size=10)
                observable_ticks = np.linspace(0, n_observables_group - 1, n_observables_group)
                plt.yticks(observable_ticks, observable_labels_ordered_group, size=10)
                outfilename = global_qa_dir / f"stat_uncertainty_prediction_{parameterization}_{group}_mean.pdf"
                plt.tight_layout()
                plt.savefig(outfilename)
                plt.close()

                # Plot ratio of relative uncertainty of prediction to that in data
                fig = plt.figure(figsize=[12, 15])
                ax = plt.axes()
                fig.suptitle(
                    "(% statistical uncertainty on prediction) / (% total uncertainty on data) -- mean", fontsize=18
                )
                matrix = np.transpose(np.mean(relative_uncertainty_ratio_to_data_group, axis=1))
                matrix_masked = np.ma.masked_where((matrix < 1e-8), matrix)
                c = ax.imshow(matrix_masked, cmap="jet", aspect="auto", vmin=0.0, vmax=5.0, interpolation="nearest")
                fig.colorbar(c)
                ax.set_xlabel("Observable bin", size=16)
                bin_ticks = range(n_bins)
                plt.xticks(bin_ticks, bin_ticks, size=10)
                observable_ticks = np.linspace(0, n_observables_group - 1, n_observables_group)
                plt.yticks(observable_ticks, observable_labels_ordered_group, size=10)
                outfilename = global_qa_dir / f"stat_uncertainty_ratio_{parameterization}_{group}_mean_0_5.pdf"
                plt.tight_layout()
                plt.savefig(outfilename)
                plt.close()

                # Repeat with different z axis range
                fig = plt.figure(figsize=[12, 15])
                ax = plt.axes()
                fig.suptitle(
                    "(% statistical uncertainty on prediction) / (% total uncertainty on data) -- mean", fontsize=18
                )
                matrix = np.transpose(np.mean(relative_uncertainty_ratio_to_data_group, axis=1))
                matrix_masked = np.ma.masked_where((matrix < 1e-8), matrix)
                c = ax.imshow(matrix_masked, cmap="jet", aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
                fig.colorbar(c)
                ax.set_xlabel("Observable bin", size=16)
                bin_ticks = range(n_bins)
                plt.xticks(bin_ticks, bin_ticks, size=10)
                observable_ticks = np.linspace(0, n_observables_group - 1, n_observables_group)
                plt.yticks(observable_ticks, observable_labels_ordered_group, size=10)
                outfilename = global_qa_dir / f"stat_uncertainty_ratio_{parameterization}_{group}_mean_0_1.pdf"
                plt.tight_layout()
                plt.savefig(outfilename)
                plt.close()

                # Plot for each design point
                plot_each_design_point = False
                for design_point_index in range(n_design_points):
                    if plot_each_design_point or design_point_index in [0]:
                        # Plot relative uncertainty of prediction
                        fig = plt.figure(figsize=[12, 15])
                        ax = plt.axes()
                        fig.suptitle(
                            f"% statistical uncertainty on prediction -- design point {design_point_index}", fontsize=24
                        )
                        matrix = np.transpose(relative_uncertainty_prediction[:, design_point_index, group_mask])
                        matrix_masked = np.ma.masked_where((matrix < 1e-8), matrix)
                        c = ax.imshow(
                            matrix_masked, cmap="jet", aspect="auto", vmin=0.0, vmax=0.2, interpolation="nearest"
                        )
                        fig.colorbar(c)
                        ax.set_xlabel("Observable bin", size=16)
                        bin_ticks = range(n_bins)
                        plt.xticks(bin_ticks, bin_ticks, size=10)
                        observable_ticks = np.linspace(0, n_observables_group - 1, n_observables_group)
                        plt.yticks(observable_ticks, observable_labels_ordered_group, size=10)
                        outfilename = (
                            global_qa_dir
                            / f"stat_uncertainty_prediction_{parameterization}_{group}_design_point{design_point_index}.pdf"
                        )
                        plt.tight_layout()
                        plt.savefig(outfilename)
                        plt.close()

                        # Plot ratio of relative uncertainty of prediction to that in data
                        fig = plt.figure(figsize=[12, 15])
                        ax = plt.axes()
                        fig.suptitle(
                            f"(% statistical uncertainty on prediction) / (% total uncertainty on data) -- design point {design_point_index}",
                            fontsize=18,
                        )
                        matrix = np.transpose(relative_uncertainty_ratio_to_data[:, design_point_index, group_mask])
                        matrix_masked = np.ma.masked_where((matrix < 1e-8), matrix)
                        c = ax.imshow(
                            matrix_masked, cmap="jet", aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest"
                        )
                        fig.colorbar(c)
                        ax.set_xlabel("Observable bin", size=16)
                        bin_ticks = range(n_bins)
                        plt.xticks(bin_ticks, bin_ticks, size=10)
                        observable_ticks = np.linspace(0, n_observables_group - 1, n_observables_group)
                        plt.yticks(observable_ticks, observable_labels_ordered_group, size=10)
                        outfilename = (
                            global_qa_dir
                            / f"stat_uncertainty_ratio_{parameterization}_{group}_design_point{design_point_index}.pdf"
                        )
                        plt.tight_layout()
                        plt.savefig(outfilename)
                        plt.close()


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from jetscape_analysis.base import helpers

    helpers.setup_logging()

    main()
