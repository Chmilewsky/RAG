GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
RESET  := $(shell tput -Txterm sgr0)

install:
	@echo "🔧 $(YELLOW)Installation...$(RESET)"
	@uv sync
	@echo "✅ $(GREEN)Environment ready !$(RESET)"
run:
	@echo "🚀 $(YELLOW)Project launch..$(RESET)"
	@uv run python3 -m src

debug:
	@echo "🐞 $(YELLOW)Debug mode enabled...$(RESET)"
	@uv run python3 -m pdb -m src

clean:
	@echo "🧹 $(YELLOW)Cleaning cache in progress...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d \( -name ".mypy_cache" -o -name ".pytest_cache" \) -exec rm -rf {} +
	@rm -rf data/output
	@echo "✨ $(GREEN)Everything is clean !$(RESET)"

fclean: clean
	@echo "🧹 $(YELLOW)Cleaning .venv in progress...$(RESET)"
	@rm -rf .venv


lint:
	@echo "🔍 $(YELLOW)analysis...$(RESET)"
	@uv run flake8 src
	@uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@echo "✅ $(GREEN)The analysis is complete !$(RESET)"

lint-strict: ## Analyse statique en mode strict
	@echo "🔍🔍🔍  $(YELLOW)strict analysis...$(RESET)"
	@uv run flake8 src
	@uv run mypy --strict src
	@echo "✅ $(GREEN)The analysis is complete!$(RESET)"

index:
	uv run python -m src index --max_chunk_size 2000

search:
	uv run python -m src search

search_dataset:
	uv run python -m src search_dataset

evalcode:
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000
evaldocs:
	./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000


.PHONY: install run debug clean fclean lint lint-strict evalcode evaldocs index search search_dataset