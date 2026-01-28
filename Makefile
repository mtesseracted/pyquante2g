# Makefile for pyquante22 project

.PHONY: clean clean-editor install test

reinstall: uninstall clean install

# Remove Python build artifacts
clean:
	@echo "Cleaning up __pycache__ and *.egg-info..."
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +

# Remove editor temporary files
scrub:
	@echo "Cleaning up editor files..."
	find . -name ".*.sw?" -type f -exec rm -f {} +
	find . -name "*~" -type f -exec rm -f {} +
	find . -name ".vscode" -type d -exec rm -rf {} +
	find . -name "*.code-workspace" -type f -exec rm -f {} +

# Editable install
install:
	pip3 install -e ".[all]""

PKG := $(notdir $(shell pwd))
uninstall:
	pip uninstall -y "$(PKG)"


# Run tests with pytest
test-all:
	pytest -m ""
