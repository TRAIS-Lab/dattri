# Makefile for running pytest locally

# Variables
PYTHON = python3
DARGLINT = darglint

# .PHONY defines parts of the makefile that are not dependent on any specific file
# This is most often used to store functions
.PHONY: setup test test-more test-full clean

# Targets
setup:
	$(PYTHON) -m pip install -e .[test,full]

test: ruff pytest

test-more: ruff pytest darglint-diff

test-full: ruff pytest darglint-full

pytest: logs
	-$(PYTHON) -m pytest -v test/ | tee test_logs/pytest.log

ruff: logs
	-$(PYTHON) -m ruff check | tee test_logs/ruff.log

darglint-diff: logs
	-$(DARGLINT) $(git diff --name-only --diff-filter=d main...HEAD -- dattri/) | tee test_logs/darglint.log

darglint-full: logs
	-$(DARGLINT) dattri/ | tee test_logs/darglint.log

logs:
	-mkdir test_logs

clean:
	-rm -rf test_logs
