[![Quality checks and Tests](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml)

# Guepard: A python library for aggregating Gaussian process sub-models

Guepard aims at building faster Gaussian process models by constructing and aggregating sub-models based on subsets of the data. It is based on GPflow and implements various aggregation methods:
* PAPL (posterior aggregation with Pseudo-Likelihood)
* more to be added!

## Install

### Using poetry

To install the library run
```
poetry install
```
in a terminal at the root of the repo

## Development
The project uses *black*, *isort*, and *flake8* for code formating and linting
```
poetry run task format
```
and it uses *pytest* for testing
```
poetry run task check
poetry run task test
```
