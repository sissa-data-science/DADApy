lint-all: black isort flake8 #vulture darglint pylint ## run all linters

isort: ## sort import statements with isort
	isort dadapy tests

isort-check: ## check import statements order with isort
	isort --check-only dadapy tests

black: ## apply black formatting
	black dadapy tests

black-check: ## check black formatting
	black --check --verbose dadapy tests

# to include the "dadapy" folder: pytest tests --doctest-modules dadapy tests/
test: ## run tests quickly with the default Python
	pytest tests --doctest-modules tests/ \
        --cov=dadapy \
        --cov-report=xml \
        --cov-report=html \
        --cov-report=term

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source dadapy --omit="*/dadapy/plot*" -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

flake8: ## check style with flake8
	flake8 dadapy/_base.py dadapy/metric_comparisons.py dadapy/clustering.py dadapy/id_estimation.py #tests dev_scripts examples

#pylint: ## run pylint
#	pylint dadapy tests
#
#vulture: ## run vulture
#	vulture dadapy dev_scripts/whitelist.py
#
#darglint: ## run vulture
#	darglint dadapy
