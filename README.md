[![Quality checks and Tests](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/NicolasDurrande/guepard/actions/workflows/quality-checks.yaml)

[![Documentation](https://github.com/NicolasDurrande/guepard/actions/workflows/docs.yaml/badge.svg)](https://github.com/NicolasDurrande/guepard/actions/workflows/docs.yaml)

# Guepard: A python library for ensembles of Gaussian process models

Guepard aims at building faster Gaussian Process (GP) models by constructing and aggregating sub-models based on subsets of the data. It is based on GPflow and implements various aggregation methods for GP ensembles:
* Equivalent Observation as described in the AISTATS submission
* Nested GPs [Rullière 2018]
* Barycenter GP [Cohen 2020]
* Several classic baselines: (generalised) Product of Expert, (robust) Bayesian committee machine, etc

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

## References

* Didier Rullière, Nicolas Durrande, François Bachoc, and Clément Chevalier. Nested Kriging predictions for datasets with a large number of observations. Statistics and Computing, 2018.
* Samuel Cohen, Rendani Mbuvha, Tshilidzi Marwala, and Marc Peter Deisenroth. Healing products of Gaussian process experts. ICML 2020.