# Makefile to automate common tasks
.PHONY: help install lint format clean setup-dev run docker-build docker-run

# Variables
PYTHON := python3
PIP := pip
BLACK := black
FLAKE8 := flake8
ISORT := isort
MYPY := mypy

help: ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === INSTALLATION ===
install: ## Install all dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pre-commit black flake8 isort bandit safety mypy
	pre-commit install

setup-dev: install-dev ## Complete development environment setup
	@echo "Development environment setup complete"
	@echo "Remember to copy .env.example to .env and configure the API keys"

# === CODE QUALITY ===
lint: ## Run code linting
	$(FLAKE8) src/ streamlit_app/ tests/ scripts/ --max-line-length=88 --extend-ignore=E203,W503
	$(MYPY) src/ --ignore-missing-imports

format: ## Format code
	$(BLACK) src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) src/ streamlit_app/ tests/ scripts/ --profile black

format-check: ## Check formatting without making changes
	$(BLACK) --check src/ streamlit_app/ tests/ scripts/ --line-length=88
	$(ISORT) --check-only src/ streamlit_app/ tests/ scripts/ --profile black

check-all: lint format-check security-check ## Run all quality checks
	@echo "All quality checks passed!"

quality-check: check-all ## Run full quality checks
	@echo "Quality check completed!"

# === INDEXING COMMANDS ===
reindex: ## Reindex all documents in data/raw/
	$(PYTHON) -c "from src.pipeline.indexer_service import DocumentIndexer; from src.utils.embedder import AdvancedEmbedder; import glob; indexer = DocumentIndexer(AdvancedEmbedder()); pdf_files = glob.glob('data/raw/*.pdf'); indexer.index_files(pdf_files, force_recreate=True); print(f'Reindexed {len(pdf_files)} files')"

# === EVALUATION COMMANDS ===
evaluate: ## Run automatic evaluation on benchmark files
	$(PYTHON) scripts/auto_evaluate_papers.py

benchmark: ## Run benchmark analysis and generate plots
	$(PYTHON) scripts/correctness_analysis.py

# === RUN COMMANDS (integrated with scripts/run.py) ===
run: ## Start the application in local mode
	$(PYTHON) scripts/run.py local

run-dev: ## Start the application in development mode
	$(PYTHON) scripts/run.py dev

# === DOCKER COMMANDS (integrated with scripts/run.py) ===
docker-build: ## Build the Docker image
	docker build -t multimodalrag .

docker-run: docker-build ## Start the application with Docker
	$(PYTHON) scripts/run.py docker

docker-compose-up: ## Start with Docker Compose
	$(PYTHON) scripts/run.py docker-compose

docker-compose-down: ## Stop Docker Compose
	docker-compose down

# === QDRANT COMMANDS ===
qdrant-start: ## Start Qdrant in Docker
	docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

qdrant-stop: ## Stop Qdrant
	docker stop qdrant && docker rm qdrant

# === UTILITY COMMANDS ===
clean: ## Clean temporary files and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf temp/
	rm -rf data/temp/

logs: ## Show application logs
	@if [ -d "logs" ]; then tail -f logs/*.log; else echo "No log files found"; fi

check-deps: ## Check dependencies
	$(PIP) check
	$(MYPY) --strict src/
update-deps: ## Update requirements.txt with current dependencies
	$(PIP) freeze > requirements.txt

security-check: ## Check for security vulnerabilities
	safety check
	bandit -r src/ streamlit_app/ scripts/ -f json

# === COMPOUND COMMANDS ===
ci: install-dev check-all ## Complete CI pipeline
	@echo "CI pipeline completed successfully!"

dev-setup: setup-dev qdrant-start ## Complete development setup (including Qdrant)
	@echo "Development environment ready!"
	@echo "Next steps:"
	@echo "   1. Copy .env.example to .env"
	@echo "   2. Configure GROQ_API_KEY in .env file"
	@echo "   3. Run 'make run' to start the application"
