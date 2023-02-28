ARCH = $(shell uname -m)
XUNIT_FILE ?= ./test-results.xml

.PHONY: install

.PHONY: format
format:
	pipenv run isort .
	pipenv run black .

.PHONY: format-check
format-check:
	pipenv run black . --check
	pipenv run isort . --diff

.PHONY: lint
lint:
	pipenv run flake8

.PHONY: install
install:

	pipenv install --dev

