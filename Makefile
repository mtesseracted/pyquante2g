# Makefile for pyquante22 project

.PHONY: clean install test

reinstall: uninstall clean install

# Remove Python build artifacts
clean:
	@echo "Cleaning up __pycache__ and *.egg-info..."
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.egg-info" -type d -exec rm -rf {} +

# Editable install
install:s
	pip3 install -e ".[all]""

PKG := $(notdir $(shell pwd))
uninstall:
	pip uninstall -y "$(PKG)"

# Run tests with pytest
test:
	pytest
