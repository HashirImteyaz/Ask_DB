# NLQ PLM System Makefile

.PHONY: help install install-dev test test-unit test-integration clean start-api start-streamlit setup lint format

# Default target
help:
	@echo "NLQ PLM System - Available Commands:"
	@echo "=================================="
	@echo "Setup & Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Complete project setup"
	@echo ""
	@echo "Running the System:"
	@echo "  start-api    - Start the FastAPI server"
	@echo "  start-streamlit - Start the Streamlit interface"
	@echo "  start-both   - Start both API and Streamlit"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo ""
	@echo "Development:"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean up generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         - Generate documentation"

# Installation targets
install:
	pip install -r src/config/requirements.txt

install-dev:
	pip install -r src/config/requirements.txt
	pip install -r src/config/streamlit_requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Setup target
setup: install-dev
	@echo "Setting up NLQ PLM System..."
	@echo "Checking database..."
	@if [ -f "data/raw/plm_updated.db" ]; then \
		echo "✅ Database found"; \
	else \
		echo "⚠️  Database not found at data/raw/plm_updated.db"; \
	fi
	@echo "Setup complete!"

# Running targets
start-api:
	@echo "Starting NLQ API Server..."
	python scripts/start_api.py

start-streamlit:
	@echo "Starting Streamlit Interface..."
	python scripts/start_streamlit.py

start-both:
	@echo "Starting both API and Streamlit..."
	@echo "Note: Run 'make start-api' in one terminal and 'make start-streamlit' in another"

# Testing targets
test:
	python -m pytest tests/ -v --cov=src

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

# Development targets
lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  - README.md (main documentation)"
	@echo "  - QUICKSTART.md (quick start guide)"
	@echo "  - docs/STREAMLIT_README.md (Streamlit guide)"

# Utility targets
check-db:
	@if [ -f "data/raw/plm_updated.db" ]; then \
		echo "✅ Database found at data/raw/plm_updated.db"; \
	else \
		echo "❌ Database not found at data/raw/plm_updated.db"; \
		echo "Please ensure the database file is in the correct location"; \
	fi

check-deps:
	@echo "Checking Python dependencies..."
	python -c "import fastapi, streamlit, langchain, pandas; print('✅ All main dependencies available')"

status:
	@echo "NLQ PLM System Status:"
	@echo "====================="
	@$(MAKE) check-db
	@$(MAKE) check-deps
	@echo ""
	@echo "Project structure:"
	@echo "  - Source code: src/"
	@echo "  - Tests: tests/"
	@echo "  - Scripts: scripts/"
	@echo "  - Data: data/"
	@echo "  - Documentation: docs/"