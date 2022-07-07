.PHONY: help install format check test

LIB_NAME = guepard
TESTS_NAME = tests


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
	pip install -r requirements.txt -r dev_requirements.txt -e .


format: ## Formats code with `black` and `isort`
	@echo "\n=== Lint =============================================="
	autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py $(LIB_NAME) $(TESTS_NAME)
	@echo "\n=== isort =============================================="
	isort .
	@echo "\n=== black =============================================="
	black $(LIB_NAME) $(TESTS_NAME)


check: ## Runs all static checks such as code formatting checks, linting, mypy
	@echo "\n=== flake8 (linting)===================================="
	flake8 --statistics --exclude=.ipynb_checkpoints $(LIB_NAME) $(TESTS_NAME)
	@echo "\n=== black (formatting) ================================="
	black --check --diff $(LIB_NAME) $(TESTS_NAME)
	@echo "\n=== isort (imports) ===================================="
	isort --check --diff .
	@echo "\n=== mypy (static type checking) ========================"
	mypy --disallow-untyped-defs $(LIB_NAME) && mypy --allow-untyped-defs $(TESTS_NAME)


test: ## Run unit and integration tests with pytest
	pytest -v -x --ff -rN -Wignore -s --tb=short --durations=10 $(TESTS_NAME)
