# Getting started with JETSCAPE-analysis

Below I attempt to outline the information needed to get up to speed on the jetscape-analysis framework. It's divided into the analysis code and the data curation.

Work is ongoing in the [dev-stat-observables](https://github.com/JETSCAPE/JETSCAPE-analysis/tree/dev-stat-observables) branch

# Essentials for running the code

Before getting to code structure, a quick primer on running the code:

## Containers

JETSCAPE-analysis relies on a number of packages that are not so trivial to install, so it's easiest to run in a container. There are two potential options:

1. Use the STAT singularity/apptainer container. This is guaranteed to work. As of 2026 Jan, the most recent container for development (e.g. mostly unoptimized, denoted as "local") is v5.2, and it can be [found here](https://cernbox.cern.ch/s/RKBPlvHr5wxe16C).
2. Use the main JETSCAPE docker container. This has not been tested by RJE. You can try, but no promises. (It worked at some point, but changes in the architecture broke it at some point - especially for folks on ARM chips, such as Macs. See some discussion [here](https://jetscapeworkspace.slack.com/archives/C0128SDRUG4/p1750462354926279)).

The following instructions are assuming that you're using option #1. Note that if you don't have apptainer or singularity available, you can also try running the container with recent versions of podman. Alternatively, try out your local cluster / HPC system - it should be available there.

Some additional info on containers is available in the private [stat-xsede repo](https://github.com/JETSCAPE/STAT-XSEDE-2021/tree/main/containers). Ask STAT conveners for help if you don't have access.

## Simulation output files

For development of the analysis code, you'll need some simulation outputs. In principle, you can generate them yourself using the container (see the notes on using the container "apps" below), but this takes additional setup. Easier is to use these examples output files:

- Small samples:
  - [pp, 5.02 TeV, 150k events](https://cernbox.cern.ch/s/WehcAM2kwuXmeTe) (400 MB)
  - [0-10% Pb-Pb, 5.02 TeV, 150k events](https://cernbox.cern.ch/s/hIYbdxkQv4XvNRa) (1.5 GB)
- Medium samples:
  - [pp, 5.02 TeV, 2M events](https://cernbox.cern.ch/s/emZRq0BzkPxcY8p) (3.6 GB)
  - [0-10% Pb-Pb, 5.02 TeV, 500k events](https://cernbox.cern.ch/s/PLVGxHsTYc7JZVA) (2.5 GB)

These samples correspond to a test production, known as the "NoRotationFix"[^0], but corresponds to approximately the same conditions as the first production. It is design point index 12 in the exponential design. (n.b. this has a very high Q0, so it may not agree with data so well). In practice, these details are included for documentation, but shouldn't make a difference for observable development.

Please extract these to their own folders - be sure to avoid name conflicts.

[^0]: It was intended to test an issue with the coordinate rotation of the soft sector. We found that it made little different. And in any case, the NoRotationFix configuration corresponds to exactly what we ran for the 2021-2022 production.

## Run the analysis

This is not a full guide to containers, but only covers essential information. I strongly recommend searching or using an LLM to help you understand the options.

> [!TIP]
> If an observable that you're not working on isn't working, try disabling it in the config, or worse case, temporarily commenting it out. This codebase is rapidly changing, and things are sometimes broken.

### Using the container apps

For "standard" development cases, you can let the container do much of the work of steering the analysis. To do so, there are two apps built into the apptainer container:

1. `simulations`: runs jetscape simulations following the provided jetscape config. See above for some sample outputs.
2. `post-processing`: runs jetscape-analysis post-processing, analyzing simulations and storing observables.

You can check this info using apptainer:

```bash
$ apptainer run-help stat_local_gcc_v5.2.sif
  Two apps are implemented:
      - `simulations`, which runs jetscape simulations following the provided jetscape config.
      - `post-processing`, which runs jetscape post-processing, analyzing simulations and storing observables.
```

For more information on the `simulations`, see the run-help, as well as the stat-xsede repo. If you're working on development, you'll need simulations output - see above.

Of most interest is the `post-processing`, which will steer the jetscape-analysis code. It will run the analysis code over your input files, producing output observable skims and histograms (for more, see below). The run-help provides info on the arguments that need to be passed to the container:

```bash
$ apptainer run-help --app post-processing stat_local_gcc_v5.2.sif
    Run post processing of jetscape simulations.

    Container v5.2

    Args:
        <analysisConfig> <sqrtS> <observableOutputDir> <histogramOutputDir> <finalStateHadrons01.parquet> [<finalStateHadrons02.parquet> ...]

    The app will create observables and histograms from the given simulation outputs.
```

Example: here the inputs are stored in `/storage/analysis_output/`, outputs are stored in `/storage/outputs`, and I use my local development version of jetscape-analysis

```bash
$ apptainer run --cleanenv --no-home -B /storage -B /my_local_dir/jetscape-analysis:/jetscapeOpt/jetscape-analysis --app post-processing containers/local/stat_local_gcc_v5.2.sif /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml 5020 /storage/analysis_output/observables /storage/analysis_output/histograms /storage/input/jetscape_pp_5020_0000_final_state_hadrons_02.parquet
```

Some common mistakes are:

- Missing input/output directories (some are created by default, but it depends on the circumstances)
- Wrong paths: note that some are container paths, and some are you system paths.

### Manually running the analysis code

Rarely, the steering script above doesn't work, or you need manual control. In that case, I list the commands that you would use to manually run the analysis.

> [!TIP]
> As a general statement, if you're manually running commands with python, it's best to run it with `python3 -m path.to.file`.

First, setup is needed. Create a shell in the container, mounting your development repository directory:

```bash
apptainer shell --no-home --cleanenv -B /my/local/path/to/jetscape-analysis/:/jetscapeOpt/jetscape-analysis /path/to/containers/local/stat_local_gcc_v5.2.sif
# And then setup the environment
. /usr/local/init/profile.sh
module use ${JS_OPT}/heppy/modules
module load heppy/1.0
```

Now, to run the analysis:

```bash
# From the root directory of the repository. In the container, this is `/jetscapeOpt/jetscape-analysis`
# Run the analysis-code, outputting the observables skim
python3 -m jetscape_analysis.analysis.analyze_events_STAT -c /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml -i /storage/input/HYBRID_PbPb_5020_0000_final_state_hadrons_00.parquet -o /storage/analysis_output/observables
# Convert the observable skim to histograms
python3 -m plot.histogram_results_STAT -c  /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml -i /storage/analysis_output/observables/jetscape_PbPb_5020_0000_observables_00.parquet -o /storage/analysis_output/histograms/
```

> [!TIP]
> You can always check the help with `python3 -m jetscape_analysis.analysis.analyze_events_STAT --help`

And then you can go onto [plotting](#histogramming-plotting-and-normalization) (don't forget to hadd histograms from different files first). An example of running the plotting is below:

```bash
# Merge histograms
$ hadd /storage/analysis_output/histograms_pp_5020.root /storage/analysis_output/histograms/*.root
# Plot histograms
python3 -m plot.plot_results_STAT -c /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml -i /storage/analysis_output/histograms_pp_5020.root -o /storage/analysis_output/plot/5020_pp
# And then for the PbPb, assuming the histograms were stored in `/storage/analysis_output/0-10/histograms_PbPb_5020.root`
# Note that it's important that the reference final is the **final** processed pp, NOT `/storage/analysis_output/histograms_pp_5020.root`
python3 -m plot.plot_results_STAT -c /jetscapeOpt/jetscape-analysis/config/STAT_5020.yaml -i /storage/analysis_output/0-10/histograms_PbPb_5020.root -o /storage/analysis_output/plot/5020_PbPb -r /storage/analysis_output/plot/5020_pp/final_results.root
```

> [!NOTE]
> Outputs per system are usually put in different directories, so the file organization in the example is overly simplified. Use the above as guidance, but it's up to you to handle file management.

# Analysis code

Analysis code structure: The main functionality is defined in two classes:

- The base class: `jetscape_analysis/analysis/analyze_events_base_STAT.py`:
- The main analysis class: `jetscape_analysis/analysis/analyze_events_STAT.py` (inherits from the base)

The code is run based on some comment line arguments - it basically initializes the class and runs over the given files. It will analyze from parquet files -> dict of observable values (sometimes called a "skim"). To analyze at scale, we just run everything in parallel.

> [!NOTE]
> The skim was used since we don't always know binning and/or were worried about binning mistakes. In practice, this seems to be less of an issue (i.e. we're more likely to fully reanalyze if there are problems). However, it's more work to remove this step than it is to live with it (although it makes EECs tricky. To be discussed as of Nov 2025)

## Base class

This is basic functionality related to defining the analysis structure. It handles:

- file IO
  - Reading the parquet file. See further details below.
  - Appropriately tracking event-level quantities (e.g. pt_hat, weight, centrality, etc), and the output observables (stored as a dictionary). These are written into a new parquet file with pandas. There are two output files:
    - observables
    - cross_section (containing event level quantities)
- shared functionality:
  - running parameters - `is_AA`, etc
  - Setup: e.g. background sub
- Selecting particles and triggers
  - `fill_fastjet_constituents`: selecting particles and converting to PseudoJets. Converting to PseudoJets was found to be the slowest part on the python side, which is why the ultimate conversion is handled in c++ through pyjetty. This makes part of this function one of the most performance sensitive.
  - `fill_X_candidates` (e.g. `fill_photon_candidates`): Select X candidates from an event based on the criteria. These are many based on PID selections. There is one for photons, z bosons, etc...
- High performance functions:
  - In the case of performance sensitive code, we either go to c++ (as in the case of creating PseudoJets), or we use numba to just-in-time compile python functions. A few of those functions are defined at the bottom of this file. We haven't done extensive profiling, so there probably are other hot spots to optimize, but we generally use our intuition for when hot loops make a difference (or it's easy to implement in numba)

The data is read in chunks of events via pandas, and then iterated over event-by-event in `analyze_event_chunks`.

## Analysis class

There are three competing interests we that designed this analysis class around:

1. Minimizing output size
2. Reducing complexity
3. Reducing computational cost
   There are loosely listed by priority - that is to say, we tried most to minimize output size. For observables, this means that the goal is to only compute the observable as often as necessary (e.g. via early centrality, sqrt_s, etc, selections), and then only store it if we have meaningful values.

To implement observables, we group them by the trigger class (inclusive is treated as a trigger class for consistency with other trigger options). In doing so, we can optimally group e.g. our jet finding together across experiments to optimize our observable computation[^1]. In terms of implementation, we have a broad collection of trigger classes (as of Fall 2025):

- Inclusive charged-particle jets. Note that these are usually referred to as "chjet"
- Inclusive full jets (please forgive the ALICE terminology, but here we've grouping calo jets from ATLAS and CMS with full jets from ALICE and STAR). Note that these are usually referred to as "jet"
- high-pt hadron
- dijet
- photon
- pion
- Z-boson

These triggers form part of a classification system.

And then within these classes, we sometimes group some sets observables - mainly in terms of groomed vs ungroomed observables due to a large number of substructure observables. In practice, try to group things together within reason.

### Code flow

The code flows as below:

Analysis of a single event is steered in `analyze_event`. Here, we:

1. Select and convert particles to PseudoJets
2. Select trigger candidates
3. Run jet finding (more below)
4. Calculate hadron observables
5. Calculate jet observables

#### 1. Select and convert particles to PseudoJets

Much of this functionality is implemented in the base class and described briefly above. Of particular note is the convention for how we define particle selections:

- status = "+" selects shower and recoil particles
- status = "-" selects hole/wake particles.
  How this is done depends on the particular model. As a reminder for JETSCAPE, shower and recoil particles have status 0, and holes have status 1. This is already encoded in the selection functionality.

These conventions are noted because they're used to encode conventions for the `PseudoJet.user_index`. The user_index is set to be (+/-) (i+1), where i is the index of the particle in the array of selected particles[^3]. Positive values denote a positive status, while negative values denote a negative status (as defined above). Since the user_index is occupied by this encoding[^4], `fill_fastjet_constituents` will also return the PID information in a separate array. Note that this array is 0 indexed, so to access it from the user_index, you must account for the indexing difference - i.e. `pid[np.abs(particle.user_index()) - 1]`.

[^3]: We need to have i+1 because we would otherwise lose the sign information for index 0. Watch out for this!
[^4]: This encoding is required for the Negative Energy Recombiner to work properly. Negative particles will be subtracted

#### 2. Select trigger candidates

The goal here is to preselect triggers to avoid having to repeatedly find the same particles. Selections done here must be consistent with all observables,, so they tend to be fairly simple - often this is a simple PID selection (but could more complex - think types of photon requirements). It may be possible to restrict these selections further, but it needs to be traded off with whether it's worth the additional effort.

Note that for historic reasons, hadron trigger candidates are treated separately. Given that they don't require any PID or other selections, this isn't so significant, but it could of course be optimized.

#### 3. Run jet finding

Jet finding is steered through the `fill_jet_observables` method. This runs both full and charged-particle jets for all jet R that are relevant for a particular sqrt_s[^5]. The jet finding itself is done in `find_jets_and_fill`. Depending on the particular trigger, some methods expect the selected jets to be fed in one at a time (e.g. `analyze_inclusive_jets`), while others (e.g. `fill_hadron_trigger_chjet_observables`) take all selected jets at once.

These methods then further break down jets by classification (e.g. groomed vs ungroomed), and finally call methods to actually calculate and store observables (called `fill_...` by convention).

[^5]: These are defined by hand in the YAML file. A nice optimization would be to go through and extract all jet R values that are defined for a particular sqrt_s YAML config

#### 4. Calculate hadron observables

Hadron observables are handled by the following methods:

- `fill_hadron_observables`
  - Single inclusive hadron observables
- `fill_hadron_correlation_observables`
  - Hadron correlations that don't have a clear e.g. trigger and recoil. For example, hadron v2 calculated with the event plane.
- `fill_hadron_trigger_hadron_observables`
  - Hadron observables that have a clear hadron trigger. For example, high-pt hadron v2 from STAR.
- `fill_pion_trigger_hadron_observables`
  - Pion triggered hadron observables
- `fill_gamma_trigger_hadron_observables`
  - Gamma triggered hadron observables
- `fill_z_trigger_hadron_observables`
  - Z triggered hadron observables

Inclusive hadron observables tend to be fairly simple. However, the other ones need more care, particularly given that we skim values. That is to say, in the case of e.g. one entry per hadron, we would end up storing a huge amount of data. Please take a look at the code to see some techniques for mitigating this issue, although there are limits to what can be optimized.

#### 5. Calculate jet observables

Jet observables are handled by a wide variety of methods (many of which are called in `analyze_inclusive_jet`):

- `fill_charged_jet_{un,}groomed_observables`
- `fill_full_jet_{un,}groomed_observables`
- `fill_hadron_trigger_chjet_observables`
- `fill_dijet_trigger_jet_observables`
- `fill_pion_trigger_chjet_observables`
- `fill_gamma_trigger_chjet_{un,}groomed_observables`
- `fill_gamma_trigger_jet_{un,}groomed_observables`
- `fill_z_trigger_jet_observables`

There's quite some variety in how jet observables are implemented. The best examples are the functions that were implemented first - i.e. `fill_charged_jet_{un,}groomed_observables` and `fill_full_jet_{un,}groomed_observables`. For details about each implementation, it's best to look at the code itself.

### Implementing an example observable

We can use CMS 5 TeV inclusive jet spectra as an example observable (it's of course already implemented).

There are three components that we need to implement:

- The yaml configuration.
- The observable implementation code
- The plotting and normalization code

#### The YAML configuration

You are probably already familiar with this from the previous data curation efforts, but it's been standardized and generalized, so there are a number of changes.

Since this observable is at 5 TeV, it would go in the `config/STAT_5020.yaml` configuration file. It's an inclusive jet measurement, so it goes under `inclusive_jet` . On that basis, we'll build up the entry below, omitting other observables for brevity. The format is described with inline comments below

```yaml
inclusive_jet:
  # By convention, RAA observables are listed at `pt`, since they're constructed using the jet spectra
  pt_cms:
    # Should the observable be enabled during calculations?
    enabled: true
    # Relevant URLs - solely for bookkeeping
    urls:
      inspire_hep: https://inspirehep.net/literature/1848440
      hepdata: https://www.hepdata.net/record/ins1848440

    # Base measurement parameters
    centrality: [[0, 10], [10, 30], [30, 50]]

    # Jet R
    # NOTE: For the sake of space for the tables, I've only included R = 0.2 and 0.4, but the full set of tables for all R is included in the STAT_5020.yaml file
    jet_R: [0.2, 0.4]
    # The overall pt range for the measurement. Should be inclusive of all measurements
    # NOTE: This can be used to implicitly cutoff a measurement. e.g. for hadron measurements, we
    #       don't to go below 5 GeV, even if the measurement goes lower.
    pt: [200., 1000.]
    # Eta cut. Could also be eta_cut_R, which does a fiducial cut (depending on the observable)
    eta_cut: 2.0

    # This will be explained below.
    data: ...
```

The above defines the main parameters of the observables. This is the most straightforward part of the observable specification. The names of the field (e.g. `centrality`, `jet_R`, etc), are standardized, and can be checked through the validation script `validate_analysis_yaml.py`.

The second half of the observable specification is the mapping from the experimental data (e.g. measured histograms) to the relevant observable. This is a simple conceptual task, but the complexity escalates quickly when covering all observables. The approach defined below is verbose, but is verifiable and much can be entered and generated through a web application, which makes it somewhat less painful.

To describe the mapping, we need to specify for each set of measured values (e.g. one histogram). We call this a HEPDataEntry, and it has to contain the following parameters:

- A full list of the parameters of the measurement.
- The name of the non-standard systematics (i.e. beyond "sys" and "stat") and how they should be mapped within our records. This is called `systematics_names`.
- Additional systematics not specified in the table itself - e.g. a global scale uncertainty that's stored in the header / comments. Here, we specify the name and the value. This is called `additional_systematics`.
- The name of the HEPdata entry (called the "table") and the index in the table (called the "index")

In the recognition that this can get tedious for every single observable, there is are two mechanism for grouping histograms together:

1. You can provide a list of values for a given parameter. For example, if the histograms are the same e.g. the pp is the same for all AA centralities, then you can just list them all at once.
2. Instead of specifying a table and index, you can provide the `combinations` key, which will then contain a list of the parameters above it in the nested hierarchy, allowing you to specify only the values that are changing. See the pp below as an example.

You can map from all measurement parameters specified above to the parameters that we need to specify below. i.e. this measurement uses `centrality`, `jet_pt`, and `jet_R`, so we have to specify all of those values below.

> [!caution]
> For technical reasons, the values provided in the config need to be serialized to string. The parsing can become much more complicated otherwise. We can relax this if needed, but we should weigh how much it really helps, especially with the web app taking care of these details.

```yaml
data:
  # We need to specify a separate block for pp and AA.
  # This is for two reasons:
  # - pp data is stored in different tables and entries than AA
  # - pp data is sometimes stored in an entirely different HEPdata record than AA
  pp:
    hepdata:
      # If it's HEPData, you must specify the record and version.
      record:
        inspire_id: 184844
        version: 1
      # Required key in pp: spectra
      # NOTE: This binning may be different than in AA. Since it has to match for the ratio, the information here is primarily used for plotting and QA
      spectra:
        # Specify the quantity that we're plotting on the x-axis
        quantity: "pt"
        # x-axis properties
        # Must specify label, and then other values are optional
        x_axis:
          label: "#it{p}_{T,jet} (GeV/#it{c})"
        # y-axis properties
        y_axis:
          label: "#frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[nb/(GeV/c)^{-1}#right]"
          range: [2e-8, 1e2]
          log: True
        # This is where we specify the correspondence between the measurement parameters
        # and the HEPdata tables.
        tables:
          # NOTE: For pp, we need to specify the centrality, since the binning can in principle
          #       vary with centrality. In this case, it's all the same, so we just list everything
          - parameters: { "centrality": ["0_10", "10_30", "30_50"] }
            systematics_names: { "TAA": "TAA", "lumi": "lumi" }
            additional_systematics: {}
            combinations:
              # This is R = 0.2, mapping to "Figure 5-1" and the 5th index (0-indexed)
              - parameters: { "jet_R": [0.2] }
                table: "Figure 5-1"
                index: 4
              # This is R = 0.4, mapping to "Figure 5-3" and the 5th index (0-indexed)
              # NOTE: The parameters here effectively expand to:
              #       parameters: {"jet_R": [0.4], "centrality": ["0_10", "10_30", "30_50"]}
              - parameters: { "jet_R": [0.4] }
                table: "Figure 5-3"
                index: 4

  AA:
    hepdata:
      record:
        inspire_id: 184844
        version: 1
      # Required key in AA: spectra
      spectra:
        # Specify the quantity that we're plotting on the x-axis
        quantity: "pt"
        x_axis:
          label: "#it{p}_{T,jet} (GeV/#it{c})"
        y_axis:
          label: "#frac{d^{2}#sigma}{d#it{p}_{T}d#it{#eta}} #left[nb/(GeV/c)^{-1}#right]"
          range: [2e-8, 1e2]
          log: True
        tables:
          # Here, we specify the R = 0.2 cases
          - parameters: { "jet_R": [0.2] }
            systematics_names: { "TAA": "TAA", "lumi": "lumi" }
            additional_systematics: {}
            # NOTE: We can specify the table here and everything below it will be
            #       use this table unless overridden.
            table: "Figure 5-1"
            combinations:
              # And then the different centralities are specified here.
              - parameters: { "centrality": ["0_10"] }
                index: 0
              - parameters: { "centrality": ["10_30"] }
                index: 1
              - parameters: { "centrality": ["30_50"] }
                index: 2
          - parameters: { "jet_R": [0.4] }
            systematics_names: { "TAA": "TAA", "lumi": "lumi" }
            additional_systematics: {}
            table: "Figure 5-3"
            combinations:
              - parameters: { "centrality": ["0_10"] }
                index: 0
              - parameters: { "centrality": ["10_30"] }
                index: 1
              - parameters: { "centrality": ["30_50"] }
                index: 2
      # Required key in AA: ratio
      # This is for the RAA.
      ratio:
        quantity: "jet_pt"
        x_axis:
          label: "#it{p}_{T,jet} (GeV/#it{c})"
        y_axis:
          label: "#it{R}_{AA}"
          y_range: [0., 1.9]
          log: True
        tables:
          - parameters: { "jet_R": [0.2] }
            systematics_names: { "TAA": "TAA", "lumi": "lumi" }
            additional_systematics: {}
            # Same exercise as above, but now we have a different figure
            table: "Figure 7-1"
            combinations:
              - parameters: { "centrality": ["0_10"] }
                index: 0
              - parameters: { "centrality": ["10_30"] }
                index: 1
              - parameters: { "centrality": ["30_50"] }
                index: 2
          - parameters: { "jet_R": [0.4] }
            systematics_names: { "TAA": "TAA", "lumi": "lumi" }
            additional_systematics: {}
            table: "Figure 7-3"
            combinations:
              - parameters: { "centrality": ["0_10"] }
                index: 0
              - parameters: { "centrality": ["10_30"] }
                index: 1
              - parameters: { "centrality": ["30_50"] }
                index: 2
    # You could also specify custom keys. e.g. `double_ratio` for the RAA double ratio.
    # You'll then be responsible for handling this properly, including with custom histogramming
    # and plot handling later.
```

If the data is not available through HEPData, we have alternative keys for where you can specify:

- Just the binning
- A `custom_data.yaml`, which can follow a custom format.
  These are to be (re)-implemented as of Dec 2025. (These were implemented in the previous version - they just need to be ported over and adapted.)

n.b. As of Dec 2025, most observable blocks have no yet been converted too this new format. The WebApp is designed to help with this. For more info, see [data curation](#data-curation)

#### Observable code implementation

Once we've define the observable, we then need to use the configuration to actually implement the measurement. The implementations are organized into functions (the structure is described [above](#5-calculate-jet-observables). Below is an excerpt of the function that is used to measure the CMS jet pt and store it in the skim.

```python
def fill_full_jet_ungroomed_observables(
    self,
    jet: PseudoJet,
    # Additional arguments are here too - I only include what is required for this discussion.
    jet_pt: float,
    jet_pt_uncorrected: float,
    jetR: float,
    jet_collection_label: str = "",
) -> None:
    """Measure and record inclusive (full) jet observables.

    Args:
        jet: Jet.
        ...
        jet_pt: (Subtracted) jet pt
        jet_pt_uncorrected: Uncorrected jet pt.
        jetR: Jet R.
        jet_collection_label: Label of the jet collection type.
    """
    # Is the observable enabled for the current event?
    # Includes shared checks for all observables, such sqrt_s, centrality, whether enabled is true, etc...
    # NOTE: We need to pass a information so we can identifier which observable we're considering.
    #       We simplify this by passing the configuration (`inclusive_jet_observables`) and
    #       the minimal identifier for the observable (`pt_cms`).
    if self.measure_observable_for_current_event(
        self.inclusive_jet_observables, observable_name="pt_cms"
    ):
        # We only want to store the jet pt within a range we can actually use
        pt_min = self.inclusive_jet_observables["pt_cms"]["pt"][0]
        pt_max = self.inclusive_jet_observables["pt_cms"]["pt"][1]
        # As well as select on the jet_R and eta
        # NOTE: By convention, we use eta_cut for not including R in the eta selection,
        #       and eta_cut_R for a fiducial selection accounting for R (e.g. eta - R)
        if (
            jetR in self.inclusive_jet_observables["pt_cms"]["jet_R"]
            and pt_min < jet_pt < pt_max
            and abs(jet.eta()) < self.inclusive_jet_observables["pt_cms"]["eta_cut"]
        ):
            # Fill the value into the output dict which contains the event-by-event skim output.
            # It is stored under an encoded name of the observable, which follows the standard
            # of {observable_class}_{observable_type}_{experiment}_{parameters}_{background_label}
            # where {parameters} is the most minimal parameters available. In this case, we
            # the centrality is already recorded as an event-by-event quantity in the skim,
            # so we only have to encode the jet_R and store the jet pt.
            # NOTE: On the background subtraction, this is particularly simple case.
            #       Here, the subtracted jet pt  is always the right value to record.
            #       In other cases, it may need conditional treatment.
            self.observable_dict_event[
                f"inclusive_jet_pt_cms_R{jetR}{jet_collection_label}"
            ].append(jet_pt)
            # For the shower recoil, it's often useful to record the unsubtracted value
            # to see that the subtract is actually working.
            if jet_collection_label in ["_shower_recoil"]:
                self.observable_dict_event[
                    f"inclusive_jet_pt_cms_R{jetR}{jet_collection_label}_unsubtracted"
                ].append(jet_pt_uncorrected)
```

#### Plotting and normalization code

The general information that you need is described in [histogramming, plotting, and normalization](#histogramming-plotting-and-normalization.

For this particular observable, I'll focus on the areas that are relevant to the CMS inclusive jet RAA. First, we'll start with the histogramming. The entry point is `histogram_results`, and it will only produce output if there are values to plot (e.g. if the skim is empty, this will just skip everything). For this observable, the histogramming itself is steering through the function `histogram_jet_observables`, which constructs the identifier for the observable based on the parameters, loads the observable info (e.g. from the HEPdata), and then calls `histogram_observable` to do the conversion of the skim into the observable. For something simple like an RAA, this is all handled automatically - you don't need to do anything further.

In the histogramming stage, you can also construct additional histograms as needed - e.g. normalization histograms or n_trig counters.

As of Dec 2025, this hasn't been fully updated for the new YAML spec, so this will need some adapting of paths. However, it should simply much of the code.

Then, onto the plotting script. This module covers both scaling the histograms as appropriate, as well as actually doing the plotting. This is steered through `plot_results`. We build up observable names in `plot_jet_observables`, then initialize and scale the histogrammed output in `init_observable`. The normalization of each observable is handled in `scale_histograms` (this could use some reorganization, since everything is crammed into one function).

In the case of `pt_cms`, the relevant section of `scale_histograms` is:

```python
if self.sqrts == 5020:
    if observable_type == "hadron":
        ...
    elif observable_type == "inclusive_jet":
        if observable == "pt_cms":
            # Scale by the eta_cut, which is updated for each observable
            h.Scale(1.0 / (2 * self.eta_cut))
            h.Scale(1.0e6)  # convert to n
```

By convention, observable properties are initialized into the main plotting class for each observable, so e.g. eta_cut will always be the eta_cut of the current observable. Most observable properties are available in this manner.

If you need to do additional post processing, such as:

- Construct a semi-inclusive / triggered observable
- Subtract histograms to calculate a difference
- etc
  this is the place to do it. See the example of the semi-inclusive hadron-jet, which is particularly involved since we need to subtract the differences of jet yields recoiling from different trigger track intervals.

As of Dec 2025, this hasn't been fully updated for the new YAML spec, so this will need some adapting of paths. However, it should simply at least some of the code.

### Analysis details

#### Background subtraction

Since it's not necessarily obvious how best to subtract the holes for every observable, we use three subtraction methods, which are described. Each observable is expected to handle **all three cases**, and needs to develop a procedure for each case. By convention, please write out how each case is handled in a comment above the observable.

1. **"Basic subtraction"**: Here, we do the regular analysis, and then develop ad-hoc procedures for each observable. It is referred to as `shower_recoil` in the code. The procedure can usually be broken down into three classes:
   1. Jet pt-like observables -- subtract holes within R
   2. Additive substructure -- subtract contributions from holes within R
   3. Non-additive substructure -- correct the jet pt only[^2]
2. **Negative energy recombiner**: Here, we do the regular analysis, but jets are clustered using the "negative-energy recombiner" (used by LBT, JETSCAPE, etc), which modifies the jet recombination procedure to subtract the four-vector of holes at each recombination step. It is referred to as `negative_recombiner` in the code. It's conceptually preferred for reclustered jet substructure observables since it takes care of the subtraction as part of the jet recombination. The procedure can be broken down into three classes:
   1. Jet pt-like observables -- no further hole subtraction
   2. Additive substructure -- subtract contributions from holes within R
   3. Non-additive substructure -- no further hole subtraction
3. **Event-wise constituent subtraction**: Holes that are spatially close to other particles are subtracted from other particles. As a choice, we use parameters that are most optimized for R = 0.4 jets. We then do the analysis as normally, and don't need to do any special treatment at the observable level. It is referred to as `constituent_subtraction` in the code.

In practice in the code, these methods are iterated over by looping over the `jet_collection_labels`.

> [!caution]
> The values in `jet_collection_labels` are expected to already have the leading underscore (i.e. `_negative_recombiner`).

[^1]: There's always room for further optimization, but this was the organizing principle
[^2]: Understood that this may not be the ideal case, but it's unclear how this may be impacting e.g. subjets, nor how it could be accounted for, so we've lived with this. The negative energy recombiner seems to be conceptually preferred for substructure, so we haven't investigated further, although it certainly could be done

# Histogramming, plotting, and normalization

To convert from the skim to the final output, we have two main steps:

- Histogramming: Convert the skims to histograms based on the observable binning. This is handled in `plot/histogram_results_stat.py`.
- Plotting: Construct the ratios, etc, and plot the histograms. This is handled in `plot/plot_results_STAT.py`

Note that it has two modes 1) without a reference file (e.g. in the case of pp), and 2) with a reference file (e.g. running PbPb and wanting to create ratios). This reference file is passed with the `-r` argument. You can always check the `--help`

Other observables which need more complex treatment - e.g. those with separate triggers, those needing random sampling, etc - can customize according to their needs in this script.

After we have creating the histogram, we need to handle the plotting. In the case of the CMS jet RAA, plotting is handed in `plot_jet_observables`. The code itself is not very complicated - it's just building up the observable string, and then the plotting itself is handled through `plot_observable`, which for AA ultimately calls `plot_RAA`.

There are opportunities for customization in each `plot_{observable_class}`, but this could be organized more effectively. As of Dec 2025, looping over parameters using the new data curation approach should simplify the generation of these parameters significantly.

# Known issues

There have been many updates, but there are also a variety of open issues. They're numbered to make it easier to discuss them, but they are not prioritized!

Conceptual:

1. [ ] How to handle energy-energy correlators, which inherently are constructed from all of the jet constituents above some low pt requirement?
   - One approach would be to effectively bin event-by-event and then merge the histogram across events. However, this is not yet supported, and should be thought through in detail
2. [ ] The new data curation method is not yet fully integrated. The building blocks for full integration are there, but this needs some further effort, particularly on the histogramming and plotting side.

Concrete:

1. [ ] 2.76 TeV Dpt_cms (i.e. fragmentation D(pt) ) is measured as a PbPb-pp difference. However, the framework currently treats it as a ratio, which gives an wrong comparison to data. This needs to be fixed (easiest is most likely in `plot_results_STAT`)

# Data curation

To ensure modularity and separation from the analysis code, I created a new module for the data curation: `jetscape_analysis/data_curation`.

Developments here are are now merged into the [dev-stat-observables](https://github.com/JETSCAPE/JETSCAPE-analysis/tree/dev-stat-observables) branch.

Since we need to switch from root to YAML files to have full access to the full set of systematic sources, there will be a ton of new files. To contain the impact of so many files on the repo (e.g. if we need to update them, etc), I've created a repo solely for the HEPData files: https://github.com/raymondEhlers/hard-sector-data-curation .

RJE will continue to document this. In the meantime, please reach out to him with any question.

Next steps:

- [ ] Connect the data curation web app to the YAML. The pieces are these, but needs follow through to fully connect with the existing module

## Data options

### Binning only

When the measured data is not available through HEPdata or in a custom format, we can fall back to just providing the axis binning.
One option is to provide bin edges, via the `bins` key.

```yaml
data:
  pp:
    # By convention, please add a comment to label what parameter this binning corresponds to.
    # e.g. for the CMS jet RAA case, we should note that this is jet pt
    bins: [100.0, 120.0, 140.0, ...]
```
