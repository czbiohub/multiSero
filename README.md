![unittests](https://github.com/czbiohub/multisero/workflows/unittests/badge.svg)
[![codecov](https://codecov.io/gh/czbiohub/multisero/branch/master/graph/badge.svg)](https://codecov.io/gh/czbiohub/multisero)

# multisero

multisero enables serological measurements with multiplexed and standard ELISA assays.

The project automates estimation of antibody titers from data collected with ELISA assays performed with [antigen-arrays](https://doi.org/10.1101/2019.12.20.885285) and [single antigens](https://doi.org/10.1101/2020.03.17.20037713).

The immediate goal is to enable specific, sensitive, and quantitative serological surveys for COVID-19. 

## Installation

On a typical Windows, Mac, or Linux computer:
* Create a conda environment: `conda create --name multisero python=3.7`
* Activate conda environment: `conda activate multisero`
* Pip install dependencies: `pip install -r requirements.txt`
* If operating from CLI, add the package to PYTHONPATH by navigating to package directory (eg: ...\serology-COVID19), and type `export PYTHONPATH=$PYTHONPATH:$(pwd)`

For installation notes for Jetson Nano, see [these notes](docs/installation.md).

## Usage
```buildoutcfg
usage: multisero.py [-h] (-e | -a) -i INPUT -o OUTPUT
                 [-wf {well_segmentation,well_crop,array_interp,array_fit}]
                 [-d] [-r] [-m METADATA]

optional arguments:
  -h, --help            show this help message and exit
  -e, --extract_od      Segment spots and compute ODs
  -a, --analyze_od      Generate OD analysis plots
  -i INPUT, --input INPUT
                        Input directory path
  -o OUTPUT, --output OUTPUT
                        Output directory path, where a timestamped subdir will
                        be generated. In case of rerun, give path to
                        timestamped run directory
  -wf {well_segmentation,well_crop,array_interp,array_fit}, --workflow {well_segmentation,well_crop,array_interp,array_fit}
                        Workflow to automatically identify and extract
                        intensities from experiment. 'Well' experiments are
                        for standard ELISA. 'Array' experiments are for ELISA
                        assays using antigen arrays printed with Scienion
                        Array Printer Default: array_fit
  -d, --debug           Write debug plots of well and spots. Default: False
  -r, --rerun           Rerun wells listed in 'rerun_wells sheets of metadata
                        file. Default: False
  -m METADATA, --metadata METADATA
                        specify the file name for the experiment metadata.
                        Assumed to be in the same directory as images.
                        Default: 'multisero_output_data_metadata.xlsx', therefore
                        metadata flag is NOT necessary when running in -e mode IF
                        metadata file has the default filename.
```
### Extract OD from antigen array images
`python multisero.py -e -i <input> -o <output>` will take metadata for antigen array and images as input, and output optical densities for each antigen.
The optical densities are stored in an excel file at the following path: `<output>/multisero_<input>_<year><month><day>_<hour><min>/median_ODs.xlsx`

If rerunning some of the wells, the input metadata file needs to contain a sheet named 'rerun_wells'
with a column named 'well_names' listing wells that will be rerun.

This [workflow](docs/workflow.md) describes the steps in the extraction of optical density.

### Generate OD analysis plots
`python multisero.py -a -i <input> -o <output> -m <METADATA>` will read multisero or scienion spot fitting outputs and generates analysis plots for each single antigen. 3 types of plots are supported for now (ROC, categorical, standard curves).
The example xlsx config file can be found in \example folder in the repo. 

An '-l' flag can be added to load the saved report from previous run to speed up loading.

### Train a classifier using information from multiple antigens
One could train a machine learning classifier using ODs from multiple antigens to potentially improve the classification accuracy for sero-positive or sero-negative. 
The following script demonstrates how to do this with xgboost tree classifiers. 
`python -m interpretation.train_classifier` 

<img src="docs/Workflow%20Schematic.png" width="600">

## Equipment list


The project aims to implement serological analysis for several antigen multiplexing approaches. 

It currently supports: 
* classical ELISA.
* antigen arrays printed with [Scienion](https://www.scienion.com/products/sciflexarrayers/).

It can be extended to support:
* antigen arrays printed with [Echo](https://www.labcyte.com/echo-liquid-handling).
* antigen multiplexing with [Luminex](https://www.luminexcorp.com/blog/multiplex-technologies-more-effective-than-elisa-for-antibody-detection/) beads. 

The antigen-arrays can be imaged with:
 * any transmission microscope with motorized XY stage.
 * turn-key plate imagers, e.g., [SciReader CL2](https://www.scienion.com/products/scireaders/).
 * Squid - a variant of [Octopi](https://www.biorxiv.org/content/10.1101/684423v1) platform from [Prakash Lab](http://web.stanford.edu/group/prakash-lab/cgi-bin/labsite/).
 
The project will also have tools for intersecting data from different assays for estimation of concentrations, determining level of cross-reactivity, ...

## Validation

Current code is validated for analysis of anigen arrays imaged with Scienion Reader and is being refined for antigen arrays imaged with motorized XY microscope and Squid.





## Contributions
We welcome bug reports, feature requests, and contributions to the code. Please see  [this](docs/contributing.md) page for most fruitful ways to contribute.

