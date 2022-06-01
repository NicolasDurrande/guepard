[![Quality checks and Tests](https://github.com/NicolasDurrande/guepard/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml)

# Guepard: A python library for aggregating Gaussian process sub-models

Guepard aims at building faster Gaussian process models by constructing and aggregading sub-models based on subsets of the data. It is based on GPflow and implements various aggregation methods:
* PAPL (posterior aggregation with Pseudo-Likelihood)
* more to be added!

## Install

### Using poetry (recommended)
To install the library run
```
poetry install
```
in a terminal at the root of the repo

### Mac OS
The following is not tested with continuous integration, but it worked in the past...
Prerequisite: Install miniforge with brew `brew install miniforge` (uninstall anaconda first if necessary)
```
conda create -n gprd
conda activate gprd
conda install python=3.8
conda install gpflow pytest jupyterlab
```
If importing gpflow in a python interpreter fails, try re-installing tensorflow
```
conda install -c apple tensorflow-deps
pip install tensorflow-macos
```

## Development
The project uses *black* and *flake8* for code formating:
```
poetry run task check
poetry run task format
```
and it uses *pytest* for testing
```
poetry run task test
```