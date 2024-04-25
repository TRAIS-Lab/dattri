# Makefile for running pytest locally

# Variables
PYTHON = python3

# .PHONY defines parts of the makefile that are not dependent on any specific file
# This is most often used to store functions
.PHONY: setup test clean

# Targets
setup:
	$(PYTHON) -m pip install -e .[test,full]

test: ruff pytest

pytest: logs
	-$(PYTHON) -m pytest -v test/ | tee test_logs/pytest.log

ruff: logs
	-$(PYTHON) -m ruff check | tee test_logs/ruff.log

logs:
	-mkdir test_logs

clean:
	-rm -rf test_logs
