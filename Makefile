.PHONY: help install format check test

LIB_NAME = gpflux
TESTS_NAME = tests

LINT_NAMES = $(LIB_NAME) $(TESTS_NAME) notebooks


help: ## Shows this help message
	# $(MAKEFILE_LIST) is set by make itself; the following parses the `target:  ## help line` format and adds color highlighting
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


install:  ## Install repo for developement
	@echo "\n=== pip install package with dev requirements =============="
	pip install 
		-r notebook_requirements.txt \
		-r tests_requirements.txt \
		tensorflow${VERSION_TF} \
		keras${VERSION_KERAS} \
		tensorflow-probability${VERSION_TFP} \
		-e .


format: ## Formats code with `black` and `isort`
	@echo "\n=== isort =============================================="
	isort .
	@echo "\n=== black =============================================="
	black --line-length=100 $(LINT_NAMES)


check: ## Runs all static checks such as code formatting checks, linting, mypy
	@echo "\n=== flake8 (linting)===================================="
	flake8 --statistics --exclude=.ipynb_checkpoints $(LINT_NAMES)
	@echo "\n=== black (formatting) ================================="
	black --check --diff $(LIB_NAME) $(TESTS_NAME)
	@echo "\n=== mypy (static type checking) ========================"
	isort --check --diff $(LIB_NAME) $(TESTS_NAME)

test: ## Run unit and integration tests with pytest
	pytest -v -x --ff -rN -Wignore -s --tb=short --durations=10 $(TESTS_NAME)
