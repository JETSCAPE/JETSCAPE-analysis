#! /bin/bash

# This script should be run to initialize the analysis environment
# if you are using the JETSCAPE docker container.

# Load heppy module
source /usr/local/init/profile.sh
# Should work for both the STAT container (where JS_OPT is defined), and will fall back to the expected /heppy/modules on the default docker container
module use ${JS_OPT}/heppy/modules
module load heppy/1.0
