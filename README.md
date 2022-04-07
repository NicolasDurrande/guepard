[![Quality checks and Tests](https://github.com/NicolasDurrande/posterior_aggregation_with_pseudo_likelihoods/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/posterior_aggregation_with_pseudo_likelihoods/actions/workflows/quality-checks.yaml)

# PAPL: posterior Aggregation with Pseudo-Likelihoods

PAPL aims at building faster Gaussian process models by constructing and aggregading sub-models based on subsets of the data.

## Install
To install the library run
```
poetry install
```
in a terminal at the root of the repo

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