[![Quality checks and Tests](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml)

# Guepard: A python library for aggregating Gaussian process sub-models

Guepard aims at building faster Gaussian process models by constructing and aggregating sub-models based on subsets of the data. It is based on GPflow and implements various aggregation methods:
* PAPL (posterior aggregation with Pseudo-Likelihood)
* more to be added!

## Install

### Using poetry (recommended)

> **_NOTE:_**  :exclamation: The poetry-based setup does *not* work on the new Apple Silicon devices. See [Installation Apple Silicon](#Installation-Apple-Silicon) for a guide on how to install on a Apple ARM machine. :exclamation:


To install the library run
```
poetry install
```
in a terminal at the root of the repo

### Installation Apple Silicon
The following is not tested with continuous integration.

(Optional) We recommend installing a virtual env. For example using `anaconda`, we create a virtual env named `gprd` using Python 3.8 as follows:
```
conda create -n gprd python=3.8
conda activate gprd
```
From now on, run all commands in the virtual env.

Install runtime and development dependencies:
```
make install
```

## Development
The project uses *black*, *isort*, and *flake8* for code formating and linting
```
make format
```
and it uses *pytest* for testing
```
make check
make test
```
