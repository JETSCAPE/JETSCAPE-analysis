This guide describes analysis and data aggregation using the jetscape frameworks. Although it's mostly general, the steps here are described particularly in the context of the hybrid-bayesian project, so if you're working on something else, you may need to adapt for your particular project.

> [!NOTE]
> This guide is written for running on the Cambridge HPC system, CSD3. It's left to the use to adapt paths, conventions, etc, as needed for your work.

# Overview

Analysis and aggregation is a multi-step process. I'll will describe each step below, but there are enough details that some useful details will likely be left out. Ask questions! At a high level, the steps are as follows:

1. Simulations + analysis (bottom left - with inputs from upper - in workflow below):
   - Part 1:
     - Forward model
     - Analysis
   - Part 2:
     - Merging (per run)
     - Aggregation (merging across runs)
2. Simulations + data (bottom left in workflow below):
   - Plotting
   - Writing out tables for Bayesian analysis
3. Bayesian analysis (bottom right in workflow below):
   - This is beyond the scope of this guide

This was nicely compiled into a workflow figure by Luna: ![[luna_workflow.pdf]]

# Bookkeeping

Before we get to the procedures, we need to discuss bookkeeping first. I know this is not the most interesting subject, but for even a project on the size of the sandbox, anything that can go wrong will go wrong at some point. If you don't carefully keep track of how you've generated files, you'll get confused. If not today, maybe in a few months. So don't skip this step as you work!

The basic unit of bookkeeping is a "Run", which corresponds to the execution of the physics model for a given set of model parameters (i.e. a design point) at a given time. Each run is assigned a unique number. The run can have many parallel jobs, but it's required to be run with the same set of parameters[^1]. These output files will then be treated identically in the analysis and the results eventually grouped together.

> [!note]
> You can also have runs for pp collisions. In that case, you often only have the one design point of your pp tune. Which is to say, you usually will have just one pp run.

[1]: Note for experts: If you want to re-run a set of parameters to e.g. increase statistics, you can either pick up after the last index of that run (e.g. if you run job index 0-199, then you could start from 200) or you can create a new run. I find the latter to be more clear conceptually, but either approach is supported by the aggregation.

By convention, run numbers are expected to have at least four digits. That is to say, when printing with e.g. printf or formatting in python, you should use something like `%04g`. As two explicit examples:

- For run 52, it should be printed as Run0052.
- For run 15023, it should be printed as Run15023.

Below is the expected structure of a run directory. For now, just keep it in mind as reference, and then we'll work through the details in the next section.

```
Run1234/
  histograms/          # Storage of processed histograms. See below
    hybrid_PbPb_Run0005_5020_0000_histograms_00.root
    hybrid_PbPb_Run0005_5020_0001_histograms_00.root

  observables/         # Storage of processed observables. See below
  hybrid_PbPb_Run0005_5020_0000_final_state_hadrons_00.parquet  # Final state hadrons from job 0000
  hybrid_PbPb_Run0005_5020_0001_final_state_hadrons_00.parquet  # Final state hadrons from job 0001
  ...
  Run1234_info.yaml    # Run info file, see below
  Run1234_config.xml   # Configuration of a single job for this run. Can be in whatever format you use to configure your simulation (xml, yaml, ini, cmnd, ...). It's optional, but super highly recommended.
```

> [!note]
> We generally expect the filenames to be of the form:
> `{model_name}_{collision_system}_{some_additional_info}_{job_index}_{file_type}_{split_index}.{ext}`
> For final state hadrons, we have:
> `hybrid_PbPb_Run0005_5020_0000_final_state_hadrons_00.parquet`, where most names are self explanatory. The "Run0005" (`some_additional_info`) is optional, and will be passed through but ignored in the processing, the "0000" is the job index, and tracks which job generated the output, and the final "00" tracks the split index, which is used when we need to split outputs into multiple parts (here, we did not choose to do so).

## Run info files

The run number identifies a unique run, but we need to store more information (i.e. metadata) about the run. For example, the values of the parameters at the given design point. We store that in a file known as a "run info" file, which by convention is named `Run${number}_info.yaml`, where `${number}` is the run number. It's stored in the base of the run output directory, as shown above. The best practice is for the run info file to be generated at the time of the simulation, since all of these parameters are known at the time of running the simulation.

An example run file is below for Run0102, which is a simulation of 0-5% PbPb at 5.02 TeV. Each field has a comments describing the purpose.

```yaml
# Calculation type:
# str: `jet_energy_loss` or `pp_baseline`
calculation_type: jet_energy_loss
# Soft sector type:
#   str: `real_time_hydro` or `precomputed_hydro` (or `N/A` for pp)
soft_sector_execution_type: precomputed_hydro
# Pythia process:
#   str: `hard_qcd`, `prompt_photon`, or `ew_boson`
pythia_process: hard_qcd
# Run number:
#   int: Run number
run_number: 102
# sqrt_s:
#   int: sqrt_s in GeV
sqrt_s: 5020
# Number of events per individual simulation job:
#   int: n_events
n_events_per_task: 5630
# Max number of events per parquet file:
#   int: n_events. We have this hard code
n_events_per_parquet_file: 6000
# Power law exponent
#   float: Value
power_law: 4.0
# Minimum pt hat
#   float: Value
min_pt_hat: 4.0
# Centrality
#   tuple(float, float): Minimum and maximum values of the centrality. Only specified for AA
centrality:
  - 0
  - 5
# Parametrization
#   dict of parametrization info. Only specified for AA. See below
parametrization:
  # Parametrization type
  #   str: value
  type: Lres-E-loss
  # Design point index
  #   int: Index of the design point run in this simulation
  design_point_index: 2
  # Parametrization values
  #   dict: str -> float, mapping from parameter names to values
  parametrization_values:
    L_res: 5.503
    kappa_sc: 0.403214
# Soft sector parameters that we don't need for hybrid-bayesian productions as of 2026 March.
# However, they're expected to have values, so we just generate them.
# We have event-by-event hydro, so we don't need this index_to_hydro_event map.
index_to_hydro_event: {}
set_reuse_hydro: null
n_reuse_hydro: null
skip_afterburner: null
number_of_repeated_sampling: null
write_qnvector: false
```

## Convention for hybrid-model sandbox, 2026

We need to define a set of Run conventions for the hybrid-model sandbox production in 2026. These choices are arbitrary, but we need to stick to them. I've made the choices of:

| System       | Run numbers |
| ------------ | ----------- |
| pp           | 1-99        |
| PbPb, 0-5%   | 100-499     |
| PbPb, 5-10%  | 500-999     |
| PbPb, 10-20% | 1000-1499   |
| PbPb, 20-30% | 1500-1999   |
| PbPb, 30-40% | 2000-2499   |
| PbPb, 40-50% | 2500-2999   |

Note that these choices are particularly convenient, since it means that once you remove the offset number, you also automatically know the design point index. That is to say, for PbPb, 5-10%, you know that Run505 uses the parameters of design point index 5[^2].

[2]: I specifically note "design point index 5" because by convention, we assume that everything is 0-indexed.

### Generating run info files after the fact

The aggregation steps below rely on so called "run info" YAML files to be available. I wrote a semi-standalone script to generate run info files after the fact: [generate_run_info.py](https://github.com/raymondEhlers/mammoth/blob/main/projects/hybrid_model/generate_run_info.py). It requires python 3.10+, and depends on numpy and pyyaml being available. It also needs the design points to be stored in `design_points/Lres-E-loss-sandbox-2026-03.dat`, in a format readable by `np.loadtxt` (the file is also [available here](https://github.com/raymondEhlers/mammoth/blob/main/projects/hybrid_model/design_points/Lres-E-loss-sandbox-2026-03.dat)).

As an example, if you wanted to run it for Run numbers 100 through 139, for centrality 0-5%, at the Cambridge facility, you could do so with:

```bash
python generate_run_info.py -f "cambridge" -r 100 140 --centrality 0 5
```

> [!warning]
> I find it's usually easier to run this on another system and then copy them over to the HPC. YMMV

Once the runinfo are generated, you'll then need to copy them into the Run directory for each run. I tend to use bash for loops, such as:

```bash
for i in Run{0100..0139}; do cp run_info/cambridge/${i}/${i}_info.yaml production_0/${i}/.; done
```

run_info files for this production were generated after the fact following the run number conventions specified. They were generated with:

```bash
# pp
python generate_run_info.py -f "cambridge" -r 1 2
# PbPb, 0-10%
python generate_run_info.py -f "cambridge" -r 100 140 --centrality 0 5
# PbPb, 5-10%
python generate_run_info.py -f "cambridge" -r 500 540 --centrality 5 10
# PbPb, 10-20%
python generate_run_info.py -f "cambridge" -r 1000 1040 --centrality 10 20
# PbPb, 20-30%
python generate_run_info.py -f "cambridge" -r 1500 1540 --centrality 20 30
# PbPb, 30-40%
python generate_run_info.py -f "cambridge" -r 2000 2040 --centrality 30 40
# PbPb, 40-50%
python generate_run_info.py -f "cambridge" -r 2500 2540 --centrality 40 50
```

## Logbook

For the aggregation to work properly, we usually need to provide a list of what runs have actually been simulated. In an analogy to an experiment, we record each simulation in a logbook, noting the conditions that were used to run it. It has some redundant info to the run info, but it useful to store in a format that's more pleasant for humans to generate and read.

An example of what this could look like is shown below (based on the 2021-2022 jetscape production):

```yaml
# The run number. This is the critical part for the aggregation to work - this key must be the run number in this format!
Run0102:
  Responsible: User A
  # Some brief summary comment
  Comment: "PbPb, 0-5% at 5020 GeV, Lres-E-loss sandbox, design point 2"
  Status: "Finished"
  # Status of the data upload to permanent storage
  Uploaded:
    final_state_hadrons_and_partons: True
    observables: True
    histograms: True
  Storage: "/path/to/main/storage/directory"
  # Jetscape ran in containers, so we recorded exactly how it wsa run here
  Software:
    Container:
      version: 3.5
      optimization: "AMD"
  # Some details on where and how it was run, for possible future analysis.
  Computing:
    Location: bridges2
    Usage: "12 hr, 552000/4800=115 CPU"
  # Computer readable specification of the parametrization
  Design:
    Type: "Lres-E-loss"
    Point: 2
  # Free form notes, such as how many jobs succeeded
  Notes: "108/115=0.94 Success"
```

You can customize this as much as you like! Record what makes sense, and leave out what doesn't. For the aggregation to work, all it needs is that:

1. There is one entry per run
2. That entry is stored under the key `Run${number}`. The contents could even by an empty map (i.e. `{}`).

The file is expected to be stored in the `${base_bookkeeping_path}/${facility}/runs.yaml`.

> [!note]
> For a production the scale of the sandbox that is completed on one system, this logbook is less critical. But if it needs to be run over multiple facilities, run by multiple users, or otherwise has discontinuities in run numbers, etc, then you need some sort of logbook to tell the aggregation what to do. This is for two reasons:
>
> 1.  If there are many runs, this can have a (very) high I/O cost to scan all directories.
> 2.  We need some way to know what runs to download from remote storage (if used).
>
> For these reasons, it's best to have a list of available runs.

# Simulations + analysis

Here, we'll briefly describe the steps to go from initial simulation to analyzed and aggregated observables. After the physics model, the `jetscape-analysis` package contains all of the functionality to go from the simulation output to files suitable for Bayesian inference analysis.

We'll use a consistent example as we run through all of the steps. Here, we are running:

- Providing the hybrid model output for design point 2 in 0-5%
- Which is translated to Run 102 in the converted output
  - The centrality 0-5% is NOT stored in the final state hadrons, so we inject it into our converted files (see the conversion below for more info). We need a value within the range, so we arbitrarily select the mean - i.e. 2.5 .
- For 200 simulation jobs indexed 0-199
- The outputs will be stored as production number 0, which is a label to keep track of the overall simulation effort
- The outputs will be stored in `/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0`, and the job files will be stored in `/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/job_scripts/`.
- It will use the STAT v6.3 container at the provided path for these jobs

> [!note]
> These options may not fully make sense yet. Stick with it as you work through part 1.

## Notes on running `jetscape-analysis`

There are a some details that you need to know to run code in jetscape-analysis. Please keep these conventions in mind as you read through the steps below.

- This is a python package, so use your usual intuition for working with python. Of particular note, the package has compiled dependencies that are rather complicated to install. You can in principle install everything into a virtualenv, but RJE highly recommends that you you use the singularity container mentioned above. It has two benefits:
  - All of the dependencies already setup
  - Loading python packages on HPC systems can become problematic due to high network storage I/O, especially for large numbers of jobs. With all of the packages loaded into the container storage, this can significantly reduce the load by loading the packages locally (which is much cheaper).
- Conventions for running:
  - Always run from the root directory of the repository
  - Invoke modules with `python3 -m path.to.module` - e.g. `python3 -m jetscape_analysis.analysis.reader.skim_ascii`. The determine these paths, convert the file path replacing `/` with `.`.
  - Most modules that you are meant to run have a `-h/--help` option to tell you which options are available. Usually, you pass some input and output options, as well the configuration file for your analysis.
  - The configuration files are in `config`, with the names denoting their source and sqrt_s - e.g. `STAT_5020.yaml`. That configuration contains all of the observables configurations, and it also specifies the sqrt_s.
- It's really useful to learn just a little bit about how to interact with apptainer/singularity containers. Consider doing a brief tutorial, chat with an LLM, etc. They're not too difficult, but some basic knowledge goes a long way to making your work easier.
  - As an example: be sure to mount the directories you need to work with using `-B`! Otherwise, you won't be able to access the simulation outputs. On the Cambridge system, it's often convenient to use `-B /rds/project/rds-hCZCEbPdvZ8`.
  - Always use `--cleanenv --no-home`. Otherwise, you risk system python installations interfering with the container. It can cause all kinds of bugs, so better to avoid it entirely.

## Part 1: Forward model + analysis

We'll use a consistent example as we work through the tasks below.

### Physics model

The physics model (sometimes called the forward model) should be run at the specified model parameters. The output of the physics model is an ASCII file containing final state hadrons. To say just a few words on expectations:

- There is functionality built into the JETSCAPE STAT .sif container to handle generation of JETSCAPE events. A bit more info on those containers can be [found here](https://github.com/JETSCAPE/JETSCAPE-analysis/blob/dev-stat-observables/docs/getting_started.md#containers). However, full instructions on using it is beyond the scope of this guide.
- For the hybrid model, we leave this to you to generate.

As noted above, you should generate run_info files when you're running each simulation to track how it was run. The files should be created per the described specification described.

For the next steps, the model outputs are expected to be **final state hadrons**. If there's any deviation from this (e.g. 2->2 scattering outgoing partons, they need to be handled explicitly **at the analysis level**).

### Output conversion

After the simulations have run, we need to convert the simulation outputs into something that is suitable for analysis. Namely, we want the outputs to be stored in compressed columnar parquet files, with the data normalized into formats we expect at the analysis level. This step effectively translates the hybrid model outputs into the JETSCAPE frameworks.

> [!warning]
> I wrote the conversion script to copy all particles into the parquet files. This means that the outgoing partons (label -2) are included in the parquet files. They are then filtered at the analysis level.

The conversion is fairly fast, but it can be fairly IO heavy with a large number of files, so it's best to use slurm (see below on how to generate them). Here, we specify conversion our hybrid simulation output file into a parquet file with the following options:

- Max 6000 events in a single parquet file. If there are more, it will split the output into multiple files, with a independent index accounting for each file.
- The analysis need to know the centrality of each event. There are a few ways to include this:
  - If it's stored in the simulation output (e.g. the event header), we can read it and store it in the parquet file
  - If it's stored in the right format, we can read it from the run info file.
  - If we know the centrality value, but it wasn't stored in the simulation output, we can inject it at this stage so that it we be stored in the parquet file. Here, we've chosen to use this option.

> [!warning]
> The injected centrality value is written event-by-event, but we can only inject a constant value, so this is a bit of a hack. It's **much** preferred to write the event-by-event values to the simulation output.

```bash
python3 -m jetscape_analysis.analysis.reader.skim_ascii \
    -i /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_2/job-2/HYBRID_Hadrons.out \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0/Run0102/HYBRID_PbPb_0002_final_state_hadrons.parquet \
    -n 6000 \
    --inject-centrality 2.5
```

Although this is fine, it's best to run in a container, so below is an example how to run arbitrary commands inside the container. Note that it's just a simple wrap of the command above.

```bash
apptainer exec --cleanenv --no-home -B /rds/project/rds-hCZCEbPdvZ8 ${container_path} bash -c "cd /jetscapeOpt/jetscape-analysis/; COLUMNS=120 python3 -m jetscape_analysis.analysis.reader.skim_ascii -i /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_2/job-2/HYBRID_Hadrons.out -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0/Run0102/HYBRID_PbPb_0002_final_state_hadrons.parquet -n 6000 --inject-centrality 2.5"
```

> [!info]
> The conversion exceptional needs to be run via this `bash -c` approach. We usually aim for a better user interface via apptainer apps, but we needed to add some command line options, so it hasn't happened yet. In the case of JETSCAPE simulations, the conversion is performed automatically, so this approach is used just for the hybrid model.

Running this will produce a single file: `HYBRID_PbPb_0002_final_state_hadrons_00.parquet`, where the latter 00 index is to account for if files are split apart.

> [!note]
> This isn't particularly complicated to run. The real effort is just keeping all of the bookkeeping and conventions consistent.

Using the slurm job generate script below, you can submit the conversion via: `sbatch submit_convert_to_parquet_prod_0_run_102.slurm`.

### Analysis

Next, we need process the parquet files into observables. This happens in two steps:

- We do the initial calculation of the observable, storing the raw value. We call this the "observables skim".
- We then histogram that observable skim, writing out a final root file.

The final output is three files: the observable skim (parquet), the cross section and other event quantities (parquet), and the processed histogram (root). This process is fairly involved, and since this guide is focused on running the analysis, we'll skip over the details.

> [!NOTE]
> If you want to learn more about implementing observables in jetscape-analysis, see [docs/getting_started.md](https://github.com/JETSCAPE/JETSCAPE-analysis/blob/dev-stat-observables/docs/getting_started.md) in the jetscape-analysis repository. (n.b. you may need to be in the dev-stat-observables branch to find it).

To run the analysis for a single output, use the container app called `post-processing`. We can see the options by asking for the app help:

```bash
$ apptainer run-help --app post-processing stat_local_gcc_v5.2.sif
    Run post processing of jetscape simulations.

    Container v6.3

    Args:
        <analysisConfig> <sqrtS> <observableOutputDir> <histogramOutputDir> --hadrons <finalStateHadrons01.parquet> [<finalStateHadrons02.parquet> ...]

    The app will create observables and histograms from the given simulation outputs.
```

An example invocation of the analysis is below:

```bash
$ apptainer exec --cleanenv --no-home \
    -B /rds/project/rds-hCZCEbPdvZ8 \
    --app post-processing \
    ${container_path} \
    /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml \
    5020 \
    /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0/Run0102/observables \
    /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0/Run0102/histograms \
    --hadrons /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0/Run0102/HYBRID_PbPb_0002_final_state_hadrons.parquet \
```

Since this analysis is performing a full analysis including jet finding under a variety of conditions, it takes a non-zero amount of time to run. e.g. ~30 min-1 hour for a typical run on hybrid model outputs. Do to this cost, the analysis should be run through slurm.

Using the slurm job generate script below, you can submit the conversion via: `sbatch submit_analysis_prod_0_run_102.slurm`.

> [!warning]
> If you wanted to modify the version of jetscape-analysis used in the container, you can either 1) commit the change to jetscape-analysis and rebuild the container (strongly preferred. See the documentation in the stat-xsede-2021 repo) or 2) mount over the jetscape-analysis directory with your jetscape-analysis dir containing the changes. In the latter case, it's your responsibility to ensure it doesn't create IO or performance problems! (n.b. these are often difficult to track and mitigate, so use this option with extreme care!)

### Running slurm jobs

Above, we discussed a number of tasks that need to be run through slurm jobs. These tasks are substantially easier to run using the STAT container ([see further details](https://github.com/JETSCAPE/JETSCAPE-analysis/blob/dev-stat-observables/docs/getting_started.md#run-the-analysis)). Nonetheless, you need to configure a number of job scripts for the various tasks. To ease this process, RJE wrote a helper script to generate job scripts for selections of runs and job indices.

The slurm job scripts can be generated using this script: [generate_job_scripts.py](https://github.com/raymondEhlers/mammoth/blob/main/projects/hybrid_model/uk_cambridge/generate_job_scripts.py). It requires python 3.10+, but otherwise should not need additional dependencies. It will generate scripts for output conversion and analysis.

We're using the same example as above, but it's reproduced here for convenience to match up with the script:

- Providing the hybrid model output for design point 2 in 0-5%
- Which is translated to Run 102 in the converted output
  - The centrality 0-5% is NOT stored in the final state hadrons, so we inject it into our converted files (see the conversion below for more info). We need a value within the range, so we arbitrarily select the mean - i.e. 2.5 .
- For 200 simulation jobs indexed 0-199
- The outputs will be stored as production number 0, which is a label to keep track of the overall simulation effort
- The outputs will be stored in `/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0`, and the job files will be stored in `/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/job_scripts/`.
- It will use the STAT v6.3 container at the provided path for these jobs

```bash
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_2 \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 2.5 \
    --production-number 0 \
    --run-number 102;
```

This will result in two files:

- `submit_convert_to_parquet_prod_0_run_102.slurm`
- `submit_analysis_prod_0_run_102.slurm`
  These files can be submitted with `sbatch`.

> [!info]
> These choices specify the conversion from Dani's production to our Run number scheme described above.

You can try `...generate_job_scripts.py --help` for more information on options. Be certain to customize these options as needed for your work.

> [!tip]
> If you specify the `hybrid-output-dir`, it will generate the conversion AND analysis slurm scripts, since that directory is needed to run the conversion. If you don't, then it will ONLY generate the analysis slurm scripts.

If you're processing over many runs, a bash for loop can be very helpful. I generate all of the job scripts for the sandbox production using:

```bash
# 0-5%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.2.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/0-5/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 2.5 \
    --production-number 0 \
    --run-number $((100 + ${i}));
done
# 5-10%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/5-10/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 7.5 \
    --production-number 0 \
    --run-number $((500 + ${i}));
done
# 10-20%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/10-20/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 15 \
    --production-number 0 \
    --run-number $((1000 + ${i}));
done
# 20-30%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/20-30/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 25 \
    --production-number 0 \
    --run-number $((1500 + ${i}));
done
# 30-40%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/30-40/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 35 \
    --production-number 0 \
    --run-number $((2000 + ${i}));
done
# 40-50%
for i in {0..39}; do
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    --hybrid-output-dir /rds/project/rds-hCZCEbPdvZ8/peibols/bayesian/runs/40-50/point_${i} \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 199 \
    --inject-centrality 45 \
    --production-number 0 \
    --run-number $((2500 + ${i}));
done
# pp
# NOTE: In the convert script (if used - it's not enabled here below...), it needs a small manual edit of PbPb -> pp in filename in the script!
# NOTE: There were only 80 jobs here.
python3 generate_job_scripts.py \
    -c /rds/project/rds-hCZCEbPdvZ8/rehlers/containers/local/stat_local_gcc_v6.3.sif \
    -o /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian \
    -s 0 \
    -e 79 \
    --production-number 0 \
    --run-number 1
```

and then a bash for loop can be convenient to submit them - e.g.:

```bash
for i in {100..139}; do echo -n "Run0${i}: "; sbatch submit_analysis_prod_0_run_${i}.slurm; done
```

## Part 2

There are two possibles approaches to address part #2 of step 1:

a) Using the steering and aggregation script built for the jetscape-analysis framework. This means you **must follow all of the conventions**.
b) Performing the steps manually. This is only possible if the bookkeeping is clean enough (e.g. no reruns of analysis that need new bookkeeping, same numbers of jobs per run, alignment of run numbers and design point indices, etc). It's still best to follow the naming conventions, even if it's not as strictly required.

I'll cover both of the workflows below:

### a) Steering and aggregation for simulations related steps

The aggregation script is stored in `plot/steer_aggregate_and_plot_observables.py`. It has a huge amount of functionality, but as of 2026 March, this script is too large for it's own good. That being said, until we find time to refactor it, we'll work with it as best we can.

#### Merging (per run)

Here, we'll merge all of the files from a given run into one file. Using our example, we'll end up with a file named: `Run0102/histograms_PbPb_Run0102_5020.root`. In addition to doing the standard aggregation script configuration (see the aggregation section), you simply need to confirm:

- That you have create the logbook described above and point the aggregation script to it.
- You need to be able to download the analysis histograms from OSN (i.e. they need to have uploaded been to OSN). If you don't know what this means, it probably means that you cannot use this method. But you can always double check with Raymond or Luna. For the hybrid-bayesian sandbox, this method **will not work**.

Assume the above conditions are met, then you just need to:

- Set `merge_histograms` to `True`

and then run as usual: `python3 -m plot.steer_aggregate_and_plot_observables`.

#### Aggregation across design points

Here, we'll merge all of the files from a particular design point into a single root file. The outcome of this will be a file in the output directory. Using our example, for PbPb this would look like: `${aggregation_dir}/histograms_aggregated/5020_PbPb_Lres-E-loss/histograms_design_point_2.root` (n.b. for pp, this would be: `${aggregation_dir}/histograms_aggregated/5020_pp/histograms.root`). In addition to doing the standard aggregation script configuration (see the aggregation section), you simply need to confirm:

- That you have create the logbook described above and point the aggregation script to it.
- You need to be able to download the analysis histograms from OSN (i.e. they need to have uploaded been to OSN). If you don't know what this means, it probably means that you cannot use this method. But you can always double check with Raymond or Luna. For the hybrid-bayesian sandbox, this method **will not work**.

Assume the above conditions are met, then you just need to:

- Set `aggregate_histograms` to `True`

and then run as usual: `python3 -m plot.steer_aggregate_and_plot_observables`.

### b) Manual aggregation for simulations related steps

It's not always convenient to run the full aggregation in different environments - e.g. on an HPC system, so below I document ways to do the first aggregation steps directly on those systems using the same container as used for analysis.

#### Merging (per run)

Once you've run the analysis, you're left with a set of histograms per job in the Run${number}/histograms directory. You need to merge these together (i.e. "per run merging") so that we can do the next steps of the aggregation. You can do this with:

```bash
# Setup
cd /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0
apptainer shell --no-home --cleanenv -B ${PWD} ../../containers/local/stat_local_gcc_v6.3.sif

# You're now inside of the container

### Below is a simple set of bash commands that you should run to do the merging inside of the container.
### You should adapt the following variables for your purposes:
### - starts: Initial run numbers to be merged. This is an array, so you can put multiple values, so shown below. Default: All runs as of 2026-03-19
### - size: Number of runs to process. If you're merging everything, this should be equal to the zero-indexed number of your design points. Default: 39 (i.e. corresponding to 40 design points), which will then process all runs from RunXX00-RunXX39.

starts=(100 500 1000 1500 2000 2500)
size=39
for start in "${starts[@]}"; do
  end=$((start + size))
  for i in $(seq -f "%04g" "$start" "$end"); do
    echo "$i"
    hadd Run${i}/histograms_PbPb_Run${i}_5020.root Run${i}/histograms/*.root
  done
done
```

#### Aggregation across design points

To aggregate all of the Pb-Pb together, used the steps below. Be certain to edit the following variables as needed:

- `production_number`: Production number.
- `parametrization_tag`: Label of the parametrization, starting with a leading `_`. n.b. you're responsible for the leading underscore to allow it to be used with a parametrization tag, if the situation arises.
- `aggregation_label`: Label this aggregation so we can keep track. RJE tends to use the data.
- `starts`: The leading two digits in the run number, which are expected to varying between datasets (e.g. centralities). Default: `(01 05 10 15 20 25)`, which corresponds to Run01XX, Run05XX, Run10XX..., covering 0-50%.

> [!note]
> You may need to adapt some of the paths for your particular case. However, it's best to use the structure within the aggregation_label directory that we define here

```bash
# Setup
cd /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian
apptainer shell --no-home --cleanenv -B ${PWD} ../containers/local/stat_local_gcc_v6.3.sif

# Setup inside container
production_number=0
parametrization_tag="_Lres-E-loss"
aggregation_label="2026-03-19"
starts=(01 05 10 15 20 25)

# Configuration inside container
cd /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/aggregation/production_${production_number}/${aggregation_label}
aggregation_base_dir="/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/aggregation/production_${production_number}/${aggregation_label}"
base_production_dir="/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_${production_number}"
mkdir -p "${aggregation_base_dir}/histograms_aggregated/5020_PbPb${parametrization_tag}"

# Merge the files together
for i in {00..39}; do
    input_files=()
    for s in "${starts[@]}"; do
        input_files+=("${base_production_dir}/Run${s}${i}/histograms_PbPb_Run${s}${i}_5020.root")
    done
    hadd "${aggregation_base_dir}/histograms_aggregated/5020_PbPb${parametrization_tag}/histograms_design_point_${i}.root" \
        "${input_files[@]}"
done
```

Since there's only one design point for the pp, we can "aggregate" it just by copying the file over and naming it appropriately.

```bash
production_number=0
aggregation_label="2026-03-19"
aggregation_base_dir="/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/aggregation/production_${production_number}/${aggregation_label}"
base_production_dir="/rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_${production_number}"
cp ${base_production_dir}/Run0001/histograms_pp_Run0001_5020.root ${aggregation_base_dir}/histograms_aggregated/5020_pp/histograms.root
```

> [!note]
> This is effectively building up the `histograms_aggregated` directory for the aggregation

# Aggregating + curating simulations + data

The data is now fully aggregated, and it's time to use it. Note that it's small enough that it can often be convenient to copy to another system that's more comfortable to work on - e.g. a non-HPC system.

From here, **you must use the jetscape-analysis framework** - there's no manual option.

## Using the aggregation script

The aggregation script is stored in `plot/steer_aggregate_and_plot_observables.py`. It has a huge amount of functionality, but as of 2026 March, this script is too large for it's own good. That being said, until we find time to refactor it, we'll work with it as best we can.

The basic concept of the aggregation script is that it will transform the calculated observables into outputs that suitable for Bayesian analysis. This is done in a number of steps, which are controlled by configuration options at the top of the file:

- `download_runinfo`: download all run_info.yaml files specified in DataManagement (runs.yaml)
- `download_histograms`: download all histogram files from OSN
- `list_paths_for_selected_design_points`: Niche option to list out the underlying files before aggregation. Can be ignored.
- `merge_histograms`: Merge all histograms for each run into a single file
- `aggregate_histograms`: Aggregate all histograms for each design point into a single file (i.e. single file per design point)
- `plot_and_save_histograms`: Plot and save all histograms
- `write_tables`: Write tables for input to Bayesian analysis
- `plot_global_QA`: plot global QA

These main steering options are the ones that you'll change the most. For the most part, advancing to the next step requires completing the preceding steps. Beyond those configuration options, you'll also need to configure:

- The `facility` where the simulations were run, as stored in the run_info
- The paths to:
  - stat-xsede-2021 (to be made optional)
  - the jetscape-analysis dir
  - local_base_outputdir, which is the aggregation directory.
- You may also want to configure:
  - `n_cores`: Set to a number you're comfortable working with.

The default value of the other options is (hopefully) set to not impact running, but you may need to look at them on occasion.

Once you run all of the steps of the aggregation, the directory structure is expected to appear as follows (many files are left out for brevity):

```
.
├── global_qa
├── histograms_aggregated
│   ├── 5020_PbPb_Lres-E-loss
│   │   ├── histograms_design_point_0.root
│   │   ├── ...
│   └── 5020_pp
│       └── histograms.root
├── plot
│   ├── 5020_PbPb_Lres-E-loss
│   │   ├── 0
│   │   ├── ...
│   └── 5020_pp
├── run_info
│   └── cambridge
│       ├── Run0001
│           └── Run0001_info.yaml
│       ├── ...
└── tables
    ├── Data
    ├── Design
    └── Prediction
```

### Setup

If you're using the full offsite storage with e.g. OSN, then you can download all of the information yourself - you just need the logbook. However, we're doing a smaller production for the hybrid-bayesian sandbox, so it's much better if we seed some information ourselves.

> [!tip]
> For the hybrid-bayesian sandbox, this setup is required, since we don't upload the outputs.

After create the base aggregation directory, we need to extract:

- the run_info: You can do this using a simple bash script, such as the one below (n.b. not tested - run with care):
  ```bash
  facility="cambridge"
  mkdir -p ${base_aggregation_dir}/run_info/${facility}
  cd /rds/project/rds-hCZCEbPdvZ8/rehlers/hybrid-bayesian/production_0
  for d in Run*; do
      mkdir -p ${base_aggregation_dir}/run_info/${facility}/${d};
      cp ${d}/${d}_info.yaml ${base_aggregation_dir}/run_info/${facility}/${d}/.;
  done
  ```
  - Alternatively, if you generated the run info manually, it's already in the right format. Just copy it to the aggregation directory as is.
- the aggregated histograms: If you follow the manual aggregation across design points described above, you'll have already done this.

## Plotting

Plotting is enabled through the configuration option `plot_and_save_histograms`. If you enable it, it will plot all of the design points that are available in the aggregated histograms.

> [!warning]
> If you did manual aggregation, the required "run info dictionary" will not yet exist. You'll need to add an option to create it yourself. Easiest is to enable `download_runinfo` and set `force_download` to False (the latter option is important if you didn't upload to a remote system, such as for hybrid-bayesian).

> [!tip]
> This will plot in parallel up to n_cores. This is particularly important for this step, since creating all of the individual plots can take significant time.

Once the plots are created, be sure to review them carefully! Each design point doesn't necessarily need to describe the data, but e.g. fluctuations can cause significant problems.

## Writing out tables for Bayesian analysis

For the Bayesian inference analysis, we don't want to work with root files. Instead, we want to work with simple data tables. To write these out, enable the `write_tables` configuration option. It will write out three sets of files:

- Data: This is the experimental data. The observable characteristics are encoded into the filename, and then each row corresponds to an e.g. pt bin. This should contain all of e.g. difference sources of systematic uncertainties
- Design: This contains the design point values for each parametrization included in the aggregation. For the hybrid-bayesian project, this is just a single set of values. Each row is a design point. The file format is defined in our file specification.
- Prediction: These are the observables calculated from the simulations. The observables characteristics are encoded into the filenames (compare to the Data), but with two files per observable: one for the central values, and one for the uncertainties (i.e. the statistical uncertainties extracted from the simulations). Each row corresponds to a pt bin, with the values in the row listed by design point (i.e. first value is design point index 0, second is design point index 1, ...).

> [!tip]
> These can be as many pt bins in the predictions as there are in data, but since we generally keep predictions greater than 5 GeV, the data may have more rows than the predictions.

With this written out files, you can then input them into the [JETSCAPE/Bayesian](https://github.com/JETSCAPE/Bayesian) codebase and perform a Bayesian analysis. Happy analyzing!
